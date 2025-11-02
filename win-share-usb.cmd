net start usbipd
usbipd list

usbipd bind --busid 12-3
usbipd bind --busid 12-4
usbipd bind --busid 14-3
REM usbipd bind --busid 13-4
usbipd bind --busid 5-7

usbipd attach --wsl --busid 12-3
usbipd attach --wsl --busid 12-4
usbipd attach --wsl --busid 14-3
REM usbipd attach --wsl --busid 13-4
usbipd attach --wsl --busid 5-7

usbipd list

echo "Now run wsl-share-usb.sh in WSL"

