
# IMPORTANT: Make sure that "Enable access for assistive devices" is checked in System Preferences>>Universal Access. It is required for the AppleScript to work. You may have to reboot after this change (it doesn't work otherwise on Mac OS X Server 10.4).

cd ~/code/ibeis/build
echo "Creating Working Files, Directory, and Variables"
ibs_source=pack.temp
ibs_working=pack.temp.dmg
ibs_title=IBEIS
ibs_applicationName=IBEIS.app
ibs_size=256000
ibs_backgroundPictureName=background.png
ibs_finalDMGName=IBEIS.dmg

# Unmount if it already exists
if [ -d /Volumes/"${ibs_title}" ]; then
	echo "IBEIS Already Mounted! Unmounting..."
    hdiutil unmount /Volumes/"${ibs_title}"
    sleep 10
fi

mkdir "${ibs_source}"
echo "Copying Application"
cp -r ../dist/"${ibs_applicationName}" "${ibs_source}"/"${ibs_applicationName}"
echo "Creating DMG"
hdiutil create -srcfolder "${ibs_source}" -volname "${ibs_title}" -fs HFS+ -fsargs "-c c=64,a=16,e=16" -format UDRW -size ${ibs_size}k "${ibs_working}"
echo "Mouting DMG"
ibs_device=$(hdiutil attach -readwrite -noverify -noautoopen ""${ibs_working}"" | egrep '^/dev/' | sed 1q | awk '{print $1}')
echo "Formatting DMG"
mkdir /Volumes/"${ibs_title}"/.background
cp ../_installers/"${ibs_backgroundPictureName}" /Volumes/"${ibs_title}"/.background/"${ibs_backgroundPictureName}"
echo '
      tell application "Finder"
       tell disk "'${ibs_title}'"
             open
             set current view of container window to icon view
             set toolbar visible of container window to false
             set statusbar visible of container window to false
             set theViewOptions to the icon view options of container window
             set arrangement of theViewOptions to not arranged
             set icon size of theViewOptions to 72
             set background picture of theViewOptions to file ".background:'${ibs_backgroundPictureName}'"
             set the bounds of container window to {400, 100, 885, 430}
             make new alias file at container window to POSIX file "/Applications" with properties {name:"Applications"}
             set position of item "'${ibs_applicationName}'" of container window to {100, 100}
             set position of item "Applications" of container window to {375, 100}
             close
             open
             update without registering applications
             delay 5
          end tell
      end tell
      ' | osascript

echo "Convert & Saving DMG"
sync
sync
hdiutil detach ${ibs_device}
hdiutil convert "${ibs_working}" -format UDZO -imagekey zlib-level=9 -o "${ibs_finalDMGName}"
echo "Removing Working Files and Directory"
rm -f "${ibs_working}"
rm -rf "${ibs_source}"
mv -f "${ibs_finalDMGName}" ../dist/"${ibs_finalDMGName}"
echo "Completed"
