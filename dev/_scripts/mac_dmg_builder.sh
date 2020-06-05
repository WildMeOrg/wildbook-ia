
# IMPORTANT: Make sure that "Enable access for assistive devices" is checked in System Preferences>>Universal Access. It is required for the AppleScript to work. You may have to reboot after this change (it doesn't work otherwise on Mac OS X Server 10.4).

cd ~/code/wbia/build
echo "Creating Working Files, Directory, and Variables"
hs_source=pack.temp
hs_working=pack.temp.dmg
hs_title=IBEIS
hs_applicationName=IBEIS.app
hs_size=256000
hs_backgroundPictureName=background.png
hs_finalDMGName=IBEIS.dmg

# Unmount if it already exists
if [ -d /Volumes/"${hs_title}" ]; then
	echo "IBEIS Already Mounted! Unmounting..."
    hdiutil unmount /Volumes/"${hs_title}"
    sleep 10
fi

mkdir "${hs_source}"
echo "Copying Application"
cp -r ../dist/"${hs_applicationName}" "${hs_source}"/"${hs_applicationName}"
echo "Creating DMG"
hdiutil create -srcfolder "${hs_source}" -volname "${hs_title}" -fs HFS+ -fsargs "-c c=64,a=16,e=16" -format UDRW -size ${hs_size}k "${hs_working}"
echo "Mouting DMG"
hs_device=$(hdiutil attach -readwrite -noverify -noautoopen ""${hs_working}"" | egrep '^/dev/' | sed 1q | awk '{print $1}')
echo "Formatting DMG"
mkdir /Volumes/"${hs_title}"/.background
cp ../_installers/"${hs_backgroundPictureName}" /Volumes/"${hs_title}"/.background/"${hs_backgroundPictureName}"
echo '
      tell application "Finder"
       tell disk "'${hs_title}'"
             open
             set current view of container window to icon view
             set toolbar visible of container window to false
             set statusbar visible of container window to false
             set theViewOptions to the icon view options of container window
             set arrangement of theViewOptions to not arranged
             set icon size of theViewOptions to 72
             set background picture of theViewOptions to file ".background:'${hs_backgroundPictureName}'"
             set the bounds of container window to {400, 100, 885, 430}
             make new alias file at container window to POSIX file "/Applications" with properties {name:"Applications"}
             set position of item "'${hs_applicationName}'" of container window to {100, 100}
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
hdiutil detach ${hs_device}
hdiutil convert "${hs_working}" -format UDZO -imagekey zlib-level=9 -o "${hs_finalDMGName}"
echo "Removing Working Files and Directory"
rm -f "${hs_working}"
rm -rf "${hs_source}"
mv -f "${hs_finalDMGName}" ../dist/"${hs_finalDMGName}"
echo "Completed"
