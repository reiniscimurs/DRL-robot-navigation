#!/usr/bin/env bash
# 安裝 gazebo_sfm_plugin (行人 Social Force Model) 到指定的 catkin workspace
#
# 用法：
#   ./install_gazebo_sfm_plugin.sh                       # 裝到 ~/plugin/sfm_ws (預設，維持原本路徑)
#   ./install_gazebo_sfm_plugin.sh ~/DRL-robot-navigation/catkin_ws   # 裝進 DRL 的 workspace
#   STAGE_ONLY=1 ./install_gazebo_sfm_plugin.sh /tmp/testws           # 只解壓+修路徑，不裝依賴不編譯
#
set -euo pipefail

WS="${1:-$HOME/plugin/sfm_ws}"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
TARBALL="${TARBALL:-$SCRIPT_DIR/gazebo_sfm_plugin.tar.gz}"
PKG="$WS/src/gazebo_sfm_plugin"
STAGE_ONLY="${STAGE_ONLY:-0}"
ROSD="${ROS_DISTRO:-noetic}"

info() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------- 0. 檢查 ----------
info "檢查環境"
[ -f "$TARBALL" ] || die "找不到壓縮檔：$TARBALL （可用 TARBALL=/path/to/gazebo_sfm_plugin.tar.gz 指定）"
echo "  tarball  : $TARBALL"
echo "  workspace: $WS"

if [ "$STAGE_ONLY" != "1" ]; then
  [ -d "/opt/ros/$ROSD" ] || die "找不到 /opt/ros/$ROSD，請先安裝 ROS 或設定 ROS_DISTRO"
  # shellcheck disable=SC1090
  source "/opt/ros/$ROSD/setup.bash"
  echo "  ROS      : $ROSD"
fi

# ---------- 1. 系統依賴 ----------
if [ "$STAGE_ONLY" != "1" ]; then
  info "安裝系統依賴 (需要 sudo)"
  sudo apt-get update
  sudo apt-get install -y build-essential cmake git libboost-thread-dev \
       "ros-$ROSD-gazebo-ros" "ros-$ROSD-gazebo-ros-pkgs" "ros-$ROSD-gazebo-plugins"
fi

# ---------- 2. lightsfm (CMakeLists 寫死到 /usr/local/include) ----------
if [ "$STAGE_ONLY" != "1" ]; then
  if [ -d /usr/local/include/lightsfm ]; then
    info "lightsfm 已安裝，略過"
  else
    info "安裝 lightsfm -> /usr/local/include/lightsfm"
    SFM_SRC="${LIGHTSFM_SRC:-$HOME/lightsfm}"
    if [ ! -d "$SFM_SRC/.git" ]; then
      git clone https://github.com/robotics-upo/lightsfm.git "$SFM_SRC"
    else
      git -C "$SFM_SRC" pull --ff-only || warn "lightsfm pull 失敗，沿用現有版本"
    fi
    if [ -f "$SFM_SRC/Makefile" ]; then
      make -C "$SFM_SRC"
      sudo make -C "$SFM_SRC" install
    else                                   # 沒有 Makefile 就走 cmake
      cmake -S "$SFM_SRC" -B "$SFM_SRC/build"
      sudo cmake --build "$SFM_SRC/build" --target install
    fi
    [ -d /usr/local/include/lightsfm ] || die "lightsfm 安裝失敗：/usr/local/include/lightsfm 不存在"
  fi
fi

# ---------- 3. 解壓到 workspace ----------
info "解壓 package 到 $WS/src"
mkdir -p "$WS/src"
if [ -e "$PKG" ]; then
  BAK="$PKG.bak.$(date +%Y%m%d%H%M%S)"
  warn "$PKG 已存在，先備份成 $BAK"
  mv "$PKG" "$BAK"
fi
tar -xzf "$TARBALL" -C "$WS/src"
[ -f "$PKG/package.xml" ] || die "解壓結果不正確，$PKG/package.xml 不存在"

# ---------- 4. 移除內嵌的 .git (避免變成 embedded git repo) ----------
if [ -d "$PKG/.git" ]; then
  info "移除內嵌的 .git"
  rm -rf "$PKG/.git"
fi

# ---------- 5. 修正 sdf 內寫死的絕對路徑 ----------
info "修正 personA/personB sdf 內的絕對 mesh 路徑 -> $PKG"
for f in "$PKG"/worlds/person*_sfm.sdf; do
  [ -f "$f" ] || continue
  sed -i -E "s|file://[^<]*/gazebo_sfm_plugin/media/|file://$PKG/media/|g" "$f"
  echo "  $(basename "$f"): $(grep -c "file://$PKG/media/" "$f") 處已更新"
done

if [ "$STAGE_ONLY" = "1" ]; then
  info "STAGE_ONLY=1，到此為止（未安裝依賴、未編譯）"
  exit 0
fi

# ---------- 6. 編譯 ----------
if [ -d "$WS/build_isolated" ] || [ -d "$WS/devel_isolated" ]; then
  BUILD_CMD=catkin_make_isolated
  SETUP="$WS/devel_isolated/setup.bash"
  SO_PATH="$WS/devel_isolated/gazebo_sfm_plugin/lib/libPedestrianSFMPlugin.so"
else
  BUILD_CMD=catkin_make
  SETUP="$WS/devel/setup.bash"
  SO_PATH="$WS/devel/lib/libPedestrianSFMPlugin.so"
fi
info "編譯 workspace ($BUILD_CMD)"
cd "$WS"
$BUILD_CMD

# ---------- 7. 驗證 ----------
info "驗證"
[ -f "$SO_PATH" ] || die "沒有產生 $SO_PATH"
echo "  OK: $SO_PATH"
# shellcheck disable=SC1090
source "$SETUP"
echo "  GAZEBO_PLUGIN_PATH = ${GAZEBO_PLUGIN_PATH:-<空>}"
echo "  GAZEBO_RESOURCE_PATH = ${GAZEBO_RESOURCE_PATH:-<空>}"

cat <<MSG

安裝完成。之後每個新終端機請先：

    source $SETUP

測試：

    roslaunch gazebo_sfm_plugin cafe.launch
    roslaunch gazebo_sfm_plugin jamesmap4_person.launch

（要永久生效可把 source 那行加進 ~/.bashrc）
MSG
