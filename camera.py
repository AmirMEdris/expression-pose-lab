"""Open a camera only after validating that it can deliver an actual frame."""
import cv2


def open_camera(index=0):
    attempts = []
    for backend, name in ((cv2.CAP_DSHOW, 'DirectShow'),
                          (cv2.CAP_MSMF, 'Media Foundation'),
                          (cv2.CAP_ANY, 'automatic')):
        capture = cv2.VideoCapture(index, backend)
        keep = False
        try:
            if not capture.isOpened():
                attempts.append(f'{name}: could not open')
                continue
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                attempts.append(f'{name}: opened but returned no frame')
                continue
            print(f'Camera {index} connected through {name}: {frame.shape[1]}x{frame.shape[0]}', flush=True)
            keep = True
            return capture, frame
        except cv2.error as error:
            attempts.append(f'{name}: {error}')
        finally:
            if not keep:
                capture.release()
    raise RuntimeError(
        f'Camera {index} could not supply a frame. '+ '; '.join(attempts) +
        '. Close other camera apps and try launch_camera.cmd from Windows. '
        'If Windows Camera works but this launch does not, the Python process may lack '
        'desktop-camera access. A different device can be selected with --camera 1.')
