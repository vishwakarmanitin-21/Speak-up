Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

strDesktop = WshShell.SpecialFolders("Desktop")
strProjectDir = fso.GetParentFolderName(WScript.ScriptFullName)

Set oShortcut = WshShell.CreateShortcut(strDesktop & "\FlowAI.lnk")
oShortcut.TargetPath = strProjectDir & "\FlowAI.vbs"
oShortcut.WorkingDirectory = strProjectDir
oShortcut.Description = "FlowAI - Voice AI Assistant"
oShortcut.IconLocation = "shell32.dll,169"
oShortcut.Save

WScript.Echo "Shortcut created on Desktop!"
