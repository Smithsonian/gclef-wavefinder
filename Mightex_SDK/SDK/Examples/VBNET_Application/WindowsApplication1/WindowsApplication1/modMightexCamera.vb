﻿Imports System
Imports System.Text
Imports System.Runtime.InteropServices

Module modMightexCamera

    'Delegate Sub MyFrameHookerDelegate(ByRef lpImageAttribute As ImageProperty, ByRef Buffer As Byte)
    Delegate Sub MyFrameHookerDelegate(ByRef lpImageAttribute As ImageProperty, ByVal Buffer As IntPtr)

    Public FrameCount As Integer
    Public Average As Integer
    Public BrightestPixel As Byte
    'Public ImageBuf() As Byte
    Public _pImage As IntPtr
    Dim CurrentDevice As Integer
    Dim WindowHandle As IntPtr

    Dim FrameHooker1 As MyFrameHookerDelegate
    Dim FrameHooker2 As MyFrameHookerDelegate

    Public Const RAWDATA_IMAGE = 0
    Public Const BMPDATA_IMAGE = 1
    Public Const NORMAL_MODE = 0
    Public Const TRIGGER_MODE = 1

    <StructLayout(LayoutKind.Explicit)> _
    Public Structure ImageProperty
        <FieldOffset(0)> Public CameraID As Integer
        <FieldOffset(4)> Public Row As Integer
        <FieldOffset(8)> Public Column As Integer
        <FieldOffset(12)> Public Bin As Integer
        <FieldOffset(16)> Public XStart As Integer
        <FieldOffset(20)> Public YStart As Integer
        <FieldOffset(24)> Public ExposureTime As Integer
        <FieldOffset(28)> Public RedGain As Integer
        <FieldOffset(32)> Public GreenGain As Integer
        <FieldOffset(36)> Public BlueGain As Integer
        <FieldOffset(40)> Public TimeStamp As Integer
        <FieldOffset(44)> Public TriggerOccurred As Integer
        <FieldOffset(48)> Public TriggerEventCount As Integer
        <FieldOffset(52)> Public UserMark As Integer
        <FieldOffset(56)> Public FrameTime As Integer
        <FieldOffset(60)> Public CCDFrequency As Integer
        <FieldOffset(64)> Public ProcessFrameType As Integer
        <FieldOffset(68)> Public tFilterAcceptForFile As Integer
    End Structure

    <DllImport("BUF_USBCCDCamera_SDK_Stdcall.dll", CallingConvention:=CallingConvention.StdCall)> _
    Function BUFCCDUSB_GetModuleNoSerialNo(ByVal DeviceID As Integer, ByVal ModuleNo As StringBuilder, ByVal SerialNo As StringBuilder) As Integer
    End Function


    'Function MTUSB_GetModuleNo(ByVal DevHandle As Integer, ByVal ModuleNo As StringBuilder) As Integer
    Declare Function BUFCCDUSB_InitDevice Lib "BUF_USBCCDCamera_SDK_Stdcall" () As Integer
    Declare Function BUFCCDUSB_UnInitDevice Lib "BUF_USBCCDCamera_SDK_Stdcall" () As Integer
    Declare Function BUFCCDUSB_AddDeviceToWorkingSet Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal DeviceID As Integer) As Integer
    Declare Function BUFCCDUSB_RemoveDeviceFromWorkingSet Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal DeviceID As Integer) As Integer

    Declare Function BUFCCDUSB_StartCameraEngine Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal ParentHandle As Integer, ByVal BitMode As Integer) As Integer
    Declare Function BUFCCDUSB_StopCameraEngine Lib "BUF_USBCCDCamera_SDK_Stdcall" () As Integer

    Declare Function BUFCCDUSB_ShowFactoryControlPanel Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal DeviceID As Integer, ByVal Title As String) As Integer
    Declare Function BUFCCDUSB_HideFactoryControlPanel Lib "BUF_USBCCDCamera_SDK_Stdcall" () As Integer
    Declare Function BUFCCDUSB_StartFrameGrab Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal TotalFrames As Integer) As Integer
    Declare Function BUFCCDUSB_StopFrameGrab Lib "BUF_USBCCDCamera_SDK_Stdcall" () As Integer
    
    'Declare Function MTUSB_SetFrameSetting Lib " MT_USBCamera_SDK_Stdcall" ( DEV_HANDLE DevHandle, PImageCtl SettingPtr);
    Declare Function BUFCCDUSB_SetCameraWorkMode Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal DeviceID As Integer, ByVal WorkMode As Integer) As Integer
    Declare Function BUFCCDUSB_SetCustomizedResolution Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal deviceID As Integer, ByVal rowSize As Integer, _
                                                                                           ByVal columnSize As Integer, ByVal bin As Integer, ByVal bufferCnt As Integer) As Integer
    Declare Function BUFCCDUSB_SetXYStart Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal DeviceID As Integer, ByVal XStart As Integer, ByVal YStart As Integer) As Integer
    Declare Function BUFCCDUSB_SetGains Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal DeviceID As Integer, ByVal RedGain As Integer, ByVal GreenGain As Integer, _
                                                                            ByVal BlueGain As Integer) As Integer
    Declare Function BUFCCDUSB_SetExposureTime Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal DeviceID As Integer, ByVal ExposureTime As Integer) As Integer
    Declare Function BUFCCDUSB_SetGamma Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal Gamma As Integer, ByVal Contrast As Integer, ByVal Brightness As Integer, _
                                                                           ByVal SharpLevel As Integer) As Integer
    Declare Function BUFCCDUSB_SetBWMode Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal BWMode As Integer, ByVal hMirror As Integer, ByVal vFlip As Integer) As Integer

    Declare Function BUFCCDUSB_InstallFrameHooker Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal FrameType As Integer, ByVal FrameHooker As MyFrameHookerDelegate) As Integer

    Declare Function BUFCCDUSB_SetGPIOConfig Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal DeviceID As Integer, ByVal ConfigByte As Byte) As Integer
    Declare Function BUFCCDUSB_SetGPIOInOut Lib "BUF_USBCCDCamera_SDK_Stdcall" (ByVal DeviceID As Integer, ByVal OutputByte As Byte, ByRef InputBytePtr As Byte) As Integer



    'Declare Sub CopyMemory Lib "kernel32" Alias "RtlMoveMemory" (ByRef Destination As Byte, ByRef Source As Byte, ByVal Length As Integer)
    'Declare Sub CopyMemory Lib "kernel32" Alias "RtlMoveMemory" (ByVal Destination As IntPtr, ByRef Source As Byte, ByVal Length As Integer)
    Declare Sub CopyMemory Lib "kernel32" Alias "RtlMoveMemory" (ByVal Destination As IntPtr, ByVal Source As IntPtr, ByVal Length As Integer)
    Declare Function PostMessage Lib "user32" Alias "PostMessageA" (ByVal hwnd As IntPtr, ByVal wMsg As Integer, ByVal wParam As Integer, ByVal lParam As Integer) As IntPtr

    '
    'Declare Function GetWindowsDirectory Lib "kernel32" Alias "GetWindowsDirectoryA" (ByVal lpBuffer As String, ByVal nSize As Integer) As Integer
    'Declare Sub GetSystemInfo Lib "kernel32" (lpSystemInfo As SystemInfo)
    'Declare Sub GlobalMemoryStatus Lib "kernel32" (lpBuffer As MEMORYSTATUS)
    'Declare Function GetVersionEx Lib "kernel32" Alias "GetVersionExA" (ByRef lpVersionInformation As OSVERSIONINFO) As Integer
    'Declare Function GetSystemMetrics Lib "User32" (ByVal nIndex As Integer) As Integer
    'Declare Function GetDeviceCaps Lib "GDI32" (ByVal hDC As Integer, ByVal nIndex As Integer) As Integer
    'Declare Function TrackPopupMenu Lib "User32" (ByVal hMenu As Integer, ByVal wFlags As Integer, ByVal X As Integer, ByVal Y As Integer, ByVal nReserved As Integer, ByVal hWnd As Integer, lpReserved As Any) As Integer
    'Declare Function GetMenu Lib "User32" (ByVal hWnd As Integer) As Integer
    'Declare Function GetSubMenu Lib "User32" (ByVal hMenu As Integer, ByVal nPos As Integer) As Integer
    'Declare Function GetDesktopWindow Lib "User32" () As Integer
    'Declare Function GetDC Lib "User32" (ByVal hWnd As Integer) As Integer
    'Declare Function ReleaseDC Lib "User32" (ByVal hWnd As Integer, ByVal hDC As Integer) As Integer
    'Declare Function BitBlt Lib "GDI32" (ByVal hDestDC As Integer, ByVal X As Integer, ByVal Y As Integer, ByVal nWidth As Integer, ByVal nHeight As Integer, ByVal hSrcDC As Integer, ByVal XSrc As Integer, ByVal YSrc As Integer, ByVal dwRop As Integer) As Integer
    'Declare Sub SetWindowPos Lib "User32" (ByVal hWnd As Integer, ByVal hWndInsertAfter As Integer, ByVal X As Integer, ByVal Y As Integer, ByVal cx As Integer, ByVal cy As Integer, ByVal wFlags As Integer)
    'Declare Function GetPrivateProfileString Lib "kernel32" Alias "GetPrivateProfileStringA" (ByVal lpApplicationName As String, lpKeyName As Any, ByVal lpDefault As String, ByVal lpRetunedString As String, ByVal nSize As Integer, ByVal lpFileName As String) As Integer
    'Declare Function GetProfileString Lib "kernel32" Alias "GetProfileStringA" (ByVal lpAppName As String, lpKeyName As Any, ByVal lpDefault As String, ByVal lpReturnedString As String, ByVal nSize As Integer) As Integer
    'Declare Function waveOutGetNumDevs Lib "winmm" () As Integer
    'Declare Function GetSystemDirectory Lib "kernel32" Alias "GetSystemDirectoryA" (ByVal lpBuffer As String, ByVal nSize As Integer) As Integer
    'Declare Function sndPlaySound Lib "winmm" Alias "sndPlaySoundA" (ByVal lpszSoundName As String, ByVal uFlags As Integer) As Integer


    Sub NoFrameHooker(ByRef lpImageAttribute As ImageProperty, ByVal Buffer As IntPtr)
        'Nothing
    End Sub

    Sub MyFrameHooker(ByRef lpImageAttribute As ImageProperty, ByVal Buffer As IntPtr)
        'Dim MyStr As String
        Dim i As Integer
        Dim TotalPixel As Integer
        Dim tmpBrightestPixel As Byte
        Dim M As New Message

        tmpBrightestPixel = 0

        FrameCount = FrameCount + 1
        'ReDim ImageBuf(lpImageAttribute.Column * lpImageAttribute.Row)

        ' We copy the image data to ImageBuf()...Note that this callback is invoked in a higher
        ' priority thread... so this callback should not be blocked or do any GUI operations
        ' (Although this example does a simple GUI showing for demo purpose)...we recommend user to put the
        ' Image data and attributes in buffer and post a message to main thread for further
        ' processing.
        'CopyMemory(ImageBuf(0), Buffer, (lpImageAttribute.Column * lpImageAttribute.Row))
        If _pImage <> IntPtr.Zero Then
            CopyMemory(_pImage, Buffer, (lpImageAttribute.Column * lpImageAttribute.Row))
        End If
        TotalPixel = 0
        'For i = 0 To (lpImageAttribute.Column * lpImageAttribute.Row) - 1
        'TotalPixel = TotalPixel + ImageBuf(i)
        'If ImageBuf(i) > tmpBrightestPixel Then tmpBrightestPixel = ImageBuf(i)
        'Next
        Average = TotalPixel / (lpImageAttribute.Column * lpImageAttribute.Row)

        BrightestPixel = tmpBrightestPixel
        'MyStr = "Frames: " & FrameCount & "   Resolution: " & Column & "x" & row & " Gain (" & lpImageAttribute.RedGain _
        '& "," & lpImageAttribute.GreenGain & "," & lpImageAttribute.BlueGain & ") PixelValue: " & Average
        'Form1.StatusLabel.Text = MyStr

        'frmMain.FrameInfoLabel.Text = "Brightest pixel: " & BrightestPixel & vbCr & "Average pixel: " & Average & vbCr & "Frames: " & FrameCount
        M.Msg = &H401   'User defined message
        M.HWnd = WindowHandle
        M.WParam = BrightestPixel
        M.LParam = FrameCount
        PostMessage(M.HWnd, M.Msg, M.WParam, M.LParam)
        'DoEvents
    End Sub



    '----- FUNCTIONS FOR STARTING AND STOPPING CAMERA -----'

    Public Sub CameraSTART()
        InitializeDevices()
        OpenDevice()
        GetModuleNo()
        GetSerialNo()
        StartCameraEngine()
        SetCameraSettings()
        InstallFrameHooker()
        StartGrab()
    End Sub

    Public Sub CameraSTOP()
        'Stops the camera
        'UninstallFrameHooker()
        StopGrab()
        StopCameraEngine()
        UninitDevices()
        MsgBox("Camera Stopped")
    End Sub



    Private Sub InitializeDevices()
        Dim TotalDevices As Integer
        Dim MyStr As String
        Dim device As Integer

        If _pImage = IntPtr.Zero Then
            _pImage = Marshal.AllocHGlobal(3 * 1392 * 1040) 'Maximum space for all resolutions (CCE-C013-U)
        End If

        WindowHandle = frmMain.WinHwnd  'It's not changed...we keep it here.
        TotalDevices = BUFCCDUSB_InitDevice
        ' Important: We should have two delegate to reference the callbacks...otherwise
        ' GC will collect the callbacks back.
        FrameHooker1 = AddressOf MyFrameHooker
        FrameHooker2 = AddressOf NoFrameHooker
        If TotalDevices > 0 Then
            For device = 1 To TotalDevices
                BUFCCDUSB_AddDeviceToWorkingSet(device)
            Next
        End If
        If TotalDevices <> 1 Then

            MyStr = ("There are " & TotalDevices & " devices.")
            MsgBox(MyStr)
        End If
    End Sub

    Private Sub OpenDevice()
        CurrentDevice = 1
    End Sub


    Private Sub GetModuleNo()
        Dim ModuleNo As New StringBuilder(32)
        Dim SerialNo As New StringBuilder(32)
        Dim Result As Integer
        'Dim block As IntPtr
        'ModuleName = Space(32)
        Result = BUFCCDUSB_GetModuleNoSerialNo(CurrentDevice, ModuleNo, SerialNo)
        'block = Marshal.AllocHGlobal(32)
        'Result = MTUSB_GetModuleNo(DeviceHandle, block.ToInt32)
        'Result = MTUSB_GetModuleNo(DeviceHandle, &H12345678)
        frmMain.ModuleNoLabel.Text = "ModuleNo.: " & ModuleNo.ToString
        'Marshal.FreeHGlobal(block)
    End Sub

    Private Sub GetSerialNo()
        Dim ModuleNo As New StringBuilder(32)
        Dim SerialNo As New StringBuilder(32)
        Dim Result As Integer
        'Dim block As IntPtr
        'ModuleName = Space(32)
        Result = BUFCCDUSB_GetModuleNoSerialNo(CurrentDevice, ModuleNo, SerialNo)
        'block = Marshal.AllocHGlobal(32)
        frmMain.SerialNoLabel.Text = "SerialNo.: " & SerialNo.ToString
    End Sub

    Private Sub StartCameraEngine()
        BUFCCDUSB_StartCameraEngine(frmMain.Handle, 8) 'frmMain.hWnd????
    End Sub

    Private Sub SetCameraSettings()
        'Dim Result As Integer
        BUFCCDUSB_SetCameraWorkMode(CurrentDevice, NORMAL_MODE)
        BUFCCDUSB_SetCustomizedResolution(CurrentDevice, 1392, 1040, 0, 4)  ' Full resolution
        BUFCCDUSB_SetGains(CurrentDevice, 14, 14, 14 )
        BUFCCDUSB_SetExposureTime(CurrentDevice, 1000)  '1000x50= 50ms
    End Sub


    Private Sub InstallFrameHooker()
        BUFCCDUSB_InstallFrameHooker(BMPDATA_IMAGE, FrameHooker1)
    End Sub

    Private Sub StartGrab()
        BUFCCDUSB_StartFrameGrab(&H8888)
    End Sub

    Private Sub UninstallFrameHooker()
        BUFCCDUSB_InstallFrameHooker(BMPDATA_IMAGE, FrameHooker2)
    End Sub

    Private Sub StopGrab()
        BUFCCDUSB_StopFrameGrab()
    End Sub

    Private Sub StopCameraEngine()
        BUFCCDUSB_StopCameraEngine()
    End Sub

    Private Sub UninitDevices()
        'If _pImage <> IntPtr.Zero Then
        'Marshal.FreeHGlobal(_pImage)
        'End If
        BUFCCDUSB_UnInitDevice()
    End Sub
End Module
