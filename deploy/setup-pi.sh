#!/usr/bin/env bash
# 在树莓派（Raspberry Pi OS Bookworm）上安装夜视望远镜软件。
# 用法：在项目目录里运行 sudo bash deploy/setup-pi.sh [热点密码]
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "请用 sudo 运行：sudo bash deploy/setup-pi.sh"
  exit 1
fi

PASSWORD="${1:-nightscope}"
if [[ ${#PASSWORD} -lt 8 ]]; then
  echo "热点密码至少 8 位"
  exit 1
fi
USER_NAME="${SUDO_USER:-pi}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "== 1/3 安装依赖 =="
apt-get update
apt-get install -y python3-picamera2 python3-gpiozero python3-numpy python3-pil ffmpeg

echo "== 2/3 开 Wi-Fi 热点（名字 NightScope，手机连上后打开 http://10.42.0.1:8080）=="
if ! nmcli -t -f NAME connection show | grep -qx NightScope; then
  nmcli connection add type wifi ifname wlan0 con-name NightScope autoconnect yes ssid NightScope \
    802-11-wireless.mode ap 802-11-wireless.band bg ipv4.method shared \
    wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$PASSWORD"
fi
nmcli connection up NightScope || echo "热点暂时没起来，重启后会自动开启"

echo "== 3/3 开机自启 =="
sed -e "s#/home/pi/home-telescope#${PROJECT_DIR}#" \
    -e "s#User=pi#User=${USER_NAME}#" \
    -e "s#/home/pi/nightscope#/home/${USER_NAME}/nightscope#" \
    "${PROJECT_DIR}/deploy/nightscope.service" > /etc/systemd/system/nightscope.service
usermod -aG video,gpio "$USER_NAME"
systemctl daemon-reload
systemctl enable nightscope

cat <<MSG

安装完成。还要手动改两处配置（见 docs/NIGHTVISION.md“软件安装”一节），改完重启：
  /boot/firmware/config.txt   加上相机的 dtoverlay（以相机模组说明为准）
  /boot/firmware/cmdline.txt  行末加上 video=HDMI-A-1:1024x768@60（目镜屏幕分辨率）

重启后手机连 Wi-Fi “NightScope”（密码 ${PASSWORD}），浏览器打开 http://10.42.0.1:8080
查看运行日志：journalctl -u nightscope -f
MSG
