#!/bin/bash

# 에러 발생 시 중단하지 않음 (유연한 실행을 위해)
set +e

# VNC 비밀번호 설정 (KasmVNC 방식)
# -u: 유저명 지정
# -w: 쓰기 권한 (제어 가능)
# -r: 읽기 권한 (보기 가능)
echo -e "${VNC_PW}\n${VNC_PW}" | vncpasswd -u ${USER} -w -r

# 기존 잠금 파일 제거 (sudo 필요할 수 있음)
sudo rm -f /tmp/.X*-lock
sudo rm -f /tmp/.X11-unix/X*

# KasmVNC 인증서 생성 (HTTPS용)
if [ ! -f $HOME/.vnc/self.pem ]; then
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout $HOME/.vnc/self.pem \
        -out $HOME/.vnc/self.pem \
        -subj "/C=KR/ST=Seoul/L=Seoul/O=TemporalVLA/OU=Org/CN=localhost"
fi

# KasmVNC 서버 시작
# - Port 8444 (HTTPS)
# - User: 현재 유저
# - Desktop: XFCE
echo "Starting KasmVNC Server on port 8444..."
vncserver -port 8444 -select-de xfce -interface 0.0.0.0

echo "-------------------------------------------------------"
echo "KasmVNC is running!"
echo "Access URL: https://<Server-IP>:8444"
echo "User: $USER"
echo "Password: $VNC_PW"
echo "-------------------------------------------------------"

# 컨테이너가 종료되지 않도록 대기
tail -f /dev/null
