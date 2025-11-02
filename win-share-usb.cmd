net start usbipd
usbipd list

usbipd bind --busid 12-3
usbipd bind --busid 12-4
usbipd bind --busid 13-3
usbipd bind --busid 13-4
usbipd bind --busid 6-7

usbipd attach --wsl --busid 12-3
usbipd attach --wsl --busid 12-4
usbipd attach --wsl --busid 13-3
usbipd attach --wsl --busid 13-4
usbipd attach --wsl --busid 6-7

usbipd list

echo "Now run wsl-share-usb.sh in WSL"

