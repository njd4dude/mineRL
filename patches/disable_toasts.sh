#!/bin/bash
# Disable Minecraft's popup "toasts" (tutorial hints, "New Recipes Unlocked", advancements) in
# MineRL's bundled Minecraft, so they don't cover part of the agent's view.
#
# Why: MineRL 1.0 never loads options.txt (GameSettings.java: "// this.loadOptions();"), so every
# Minecraft starts with the new-player tutorial and there is no setting to turn it off.
# All popups go through ToastGui.add(), so this patch makes that method ignore new toasts,
# then rebuilds MineRL's Minecraft jar.
#
#   bash patches/disable_toasts.sh            # apply (safe to run again)
#   bash patches/disable_toasts.sh --undo     # restore the original jar and source
#
# Run it again after reinstalling MineRL. Stop any running Minecraft/training first.
set -euo pipefail

cd "$(dirname "$0")/.."
MCP=.venv/lib/python3.10/site-packages/minerl/MCP-Reborn
SRC=$MCP/src/main/java/net/minecraft/client/gui/toasts/ToastGui.java
JAR=$MCP/build/libs/mcprec-6.13.jar
MARK="// [patched] toasts disabled"

if [ "${1:-}" = "--undo" ]; then
    [ -f "$JAR.orig" ] && cp "$JAR.orig" "$JAR" && echo "Restored original jar"
    [ -f "$SRC.orig" ] && cp "$SRC.orig" "$SRC" && echo "Restored original ToastGui.java"
    exit 0
fi

if pgrep -f "^java .*mcprec" > /dev/null; then
    echo "Minecraft is running. Stop training/viewers first." >&2
    exit 1
fi

# 1. Back up the originals (only the first time)
[ -f "$JAR.orig" ] || cp "$JAR" "$JAR.orig"
[ -f "$SRC.orig" ] || cp "$SRC" "$SRC.orig"

# 2. Patch ToastGui.add() to return immediately
if grep -q "$MARK" "$SRC"; then
    echo "ToastGui.java already patched"
else
    python3 - "$SRC" "$MARK" <<'EOF'
import sys
path, mark = sys.argv[1], sys.argv[2]
src = open(path).read()
old = "   public void add(IToast toastIn) {\n      this.toastsQueue.add(toastIn);\n   }"
new = "   public void add(IToast toastIn) {\n      " + mark + "\n   }"
assert src.count(old) == 1, "ToastGui.add() not found as expected; MineRL version may differ"
open(path, "w").write(src.replace(old, new))
print("Patched ToastGui.add()")
EOF
fi

# 3. Rebuild the jar with Java 8 (what MineRL's Minecraft needs)
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
unset PYTHONPATH
(cd "$MCP" && ./gradlew shadowJar --offline)
echo "Done. Rebuilt $JAR"
