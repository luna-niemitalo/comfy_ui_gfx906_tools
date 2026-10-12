#!/usr/bin/env nu
# Deploys rvii-comfy into a ComfyUI checkout (DESIGN.md has the invariants):
#   1. core patches (comfyui/patches) applied with git am onto the checkout -- skipped when present
#   2. third-party nodes (third_party/nodes.toml) cloned at their pinned commit + our patches
#   3. links: our nodes, develop/, the root tooling (run.nu, nix/, .envrc, CLAUDE.md, ...) -> this repo
#   4. --oc: the OC content repo linked in as develop/succubus_oc, its tools/ -> oc_tools/
# Never deletes: a real file or dir where a link should go is left alone (reported), or with
# --replace moved to <comfy>/.rvii-backup/<timestamp>/. Re-running is safe: done steps report "ok".
#
#   nu deploy.nu <comfyui checkout> [--dry-run] [--replace] [--oc <succubus-oc repo>]

const LINKS = [
    [link target];
    ["custom_nodes/ComfyUI-Gfx906Int4" "nodes/ComfyUI-Gfx906Int4"]
    ["custom_nodes/ComfyUI-ResolutionGrowth" "nodes/ComfyUI-ResolutionGrowth"]
    ["custom_nodes/ComfyUI_Luna_QOL_Nodes" "nodes/ComfyUI_Luna_QOL_Nodes"]
    ["develop" "develop"]
    ["CLAUDE.md" "CLAUDE.md"]
    ["run.nu" "comfyui/root/run.nu"]
    ["nix" "comfyui/root/nix"]
    [".envrc" "comfyui/root/.envrc"]
    [".python-version" "comfyui/root/.python-version"]
    ["requirements_additional.txt" "comfyui/root/requirements_additional.txt"]
    ["gpu_profile.py" "comfyui/root/gpu_profile.py"]
]
# runtime data inside a dir that becomes a link: moved into this repo (gitignored) before the dir is
# backed up, so experiment outputs stay where the code expects them
const MIGRATE = [
    [from to];
    ["develop/out" "develop/out"]
]

# --wrapped: git's own flags (--check, --reverse, ...) pass through instead of parsing as ours
def --wrapped git-ok [comfy: path, ...args]: nothing -> bool {
    (^git -C $comfy ...$args | complete).exit_code == 0
}

def act [dry: bool, what: string, action: closure] {
    print $"  ($what)"
    if not $dry { do $action }
}

# IN, critical: the target must be a ComfyUI git checkout that contains the pinned upstream base
def check-checkout [comfy: path, base: string] {
    if ($comfy | path type) != "dir" {
        error make {msg: $"expected a ComfyUI checkout directory, got ($comfy) \(type: ($comfy | path type)\)"}
    }
    if not ($comfy | path join "comfy" "model_management.py" | path exists) {
        error make {msg: $"expected ($comfy)/comfy/model_management.py \(a ComfyUI tree\), not found"}
    }
    if not (git-ok $comfy rev-parse --git-dir) {
        error make {msg: $"expected ($comfy) to be a git checkout, git rev-parse failed"}
    }
    if not (git-ok $comfy merge-base --is-ancestor $base HEAD) {
        let head = (^git -C $comfy rev-parse HEAD | str trim)
        error make {msg: $"expected HEAD to contain upstream base ($base) \(comfyui/BASE\), HEAD is ($head)"}
    }
}

def core-patches [repo: path, comfy: path, dry: bool] {
    print "core patches:"
    for patch in (ls ($repo | path join "comfyui" "patches" "*.patch" | into glob) | get name | sort) {
        let name = ($patch | path basename)
        if (git-ok $comfy apply --reverse --check $patch) {
            print $"  ok ($name)"
        } else if (git-ok $comfy apply --check $patch) {
            act $dry $"git am ($name)" { ^git -C $comfy am --3way $patch }
        } else {
            error make {msg: $"($name): neither applies nor is already applied on ($comfy) -- resolve by hand \(git -C ($comfy) am --3way ($patch)\)"}
        }
    }
}

def third-party [repo: path, comfy: path, dry: bool] {
    print "third-party nodes:"
    for node in (open ($repo | path join "third_party" "nodes.toml") | get node) {
        let dir = ($comfy | path join "custom_nodes" $node.name)
        if not ($dir | path exists) {
            act $dry $"clone ($node.name) @ ($node.commit | str substring 0..9)" {
                ^git clone -q $node.url $dir
                ^git -C $dir checkout -q $node.commit
            }
        } else {
            let head = (^git -C $dir rev-parse HEAD | str trim)
            if $head != $node.commit {
                print $"  note ($node.name): pinned ($node.commit | str substring 0..9), checkout at ($head | str substring 0..9) -- left as is"
            }
        }
        let patch = ($repo | path join "third_party" "patches" $"($node.name).patch")
        if not ($patch | path exists) { continue }
        if ($dir | path exists) and (git-ok $dir apply --reverse --check $patch) {
            print $"  ok ($node.name) patch"
        } else {
            act $dry $"apply ($node.name).patch" { ^git -C $dir apply $patch }
        }
    }
}

def place-link [repo: path, comfy: path, link: string, target: path, dry: bool, replace: bool, backup: path] {
    let at = ($comfy | path join $link)
    let kind = ($at | path type)
    if $kind == "symlink" and ((^readlink -f $at | str trim) == ($target | path expand)) {
        print $"  ok ($link)"
        return
    }
    if $kind == "symlink" {
        error make {msg: $"($link): expected a link to ($target), it links to (^readlink $at | str trim) -- fix by hand"}
    }
    if ($at | path exists) {
        if not $replace {
            print $"  skip ($link): a real ($kind) is there \(--replace moves it to .rvii-backup/\)"
            return
        }
        let keep = ($backup | path join $link)
        act $dry $"back up ($link) -> ($keep)" {
            mkdir ($keep | path dirname)
            mv $at $keep
        }
    }
    act $dry $"link ($link) -> ($target)" { ^ln -s $target $at }
}

def migrate [repo: path, comfy: path, dry: bool] {
    for m in $MIGRATE {
        let from = ($comfy | path join $m.from)
        let to = ($repo | path join $m.to)
        # only out of a real old dir, never through an existing link (that would move it onto itself)
        if ($from | path dirname | path type) != "dir" or not ($from | path exists) { continue }
        if ($to | path exists) {
            print $"  note ($m.from): ($to) exists already -- not merged, both kept"
            continue
        }
        act $dry $"move ($m.from) -> ($to)" { mv $from $to }
    }
}

def oc-links [repo: path, oc: path, dry: bool] {
    print "OC content:"
    if not ($oc | path join "canon_from_chat.md" | path exists) {
        error make {msg: $"--oc: expected the OC content repo \(with canon_from_chat.md\), got ($oc)"}
    }
    let tools = ($oc | path join "tools")
    if ($tools | path exists) and ($tools | path type) != "symlink" {
        error make {msg: $"($tools): expected a link to oc_tools/ or nothing, found a real ($tools | path type)"}
    }
    if not ($tools | path exists) {
        act $dry $"link ($tools) -> oc_tools" { ^ln -s ($repo | path join "oc_tools") $tools }
    } else {
        print "  ok tools"
    }
    let at = ($repo | path join "develop" "succubus_oc")
    if ($at | path exists) {
        print "  ok develop/succubus_oc"
    } else {
        act $dry $"link develop/succubus_oc -> ($oc)" { ^ln -s ($oc | path expand) $at }
    }
}

def main [comfy: path, --dry-run, --replace, --oc: path] {
    let repo = $env.FILE_PWD
    let comfy = ($comfy | path expand)
    let base = (open ($repo | path join "comfyui" "BASE") | str trim)
    let backup = ($comfy | path join ".rvii-backup" (date now | format date "%Y%m%d_%H%M%S"))
    check-checkout $comfy $base
    print $"deploy ($repo) -> ($comfy)(if $dry_run { ' (dry run)' } else { '' })"
    core-patches $repo $comfy $dry_run
    third-party $repo $comfy $dry_run
    if $replace {
        print "migrate:"
        migrate $repo $comfy $dry_run
    }
    print "links:"
    for l in $LINKS {
        place-link $repo $comfy $l.link ($repo | path join $l.target) $dry_run $replace $backup
    }
    if $oc != null { oc-links $repo ($oc | path expand) $dry_run }
}
