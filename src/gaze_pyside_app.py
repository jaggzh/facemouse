#!/usr/bin/env python3

import sys
import time

import av  # pip install av
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

import camsettingsh264 as camsettings
import cv2
import mediapipe as mp


class VideoWorker(QtCore.QObject):
    frameReady = QtCore.Signal(np.ndarray)
    error = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )


    @QtCore.Slot()
    def start(self):
        self._running = True

        host = camsettings.camhost
        port = camsettings.camport
        url = f"tcp://{host}:{port}"

        while self._running:
            try:
                print(f"[Worker] Opening stream {url} ...")
                container = av.open(
                    url,
                    options={
                        "fflags": "nobuffer",
                        "flags": "low_delay",
                        "probesize": "32",
                        "analyzeduration": "0",
                    },
                    timeout=5.0,
                )
            except av.AVError as e:
                msg = f"Failed to open stream: {e}"
                print("[Worker]", msg)
                self.error.emit(msg)
                # back off a bit then retry
                time.sleep(1.0)
                continue

            print("[Worker] Stream opened, starting decode loop")

            try:
                for frame in container.decode(video=0):
                    if not self._running:
                        break

                    img = frame.to_ndarray(format="bgr24")

                    try:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        results = self.face_mesh.process(rgb)

                        if results.multi_face_landmarks:
                            h, w, _ = img.shape
                            for face_landmarks in results.multi_face_landmarks:
                                for lm in face_landmarks.landmark:
                                    x = int(lm.x * w)
                                    y = int(lm.y * h)
                                    cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
                    except Exception as e:
                        self.error.emit(f"Mediapipe error: {e}")

                    self.frameReady.emit(img)

            except av.AVError as e:
                msg = f"Decode error or stream ended: {e}"
                print("[Worker]", msg)
                self.error.emit(msg)

            finally:
                try:
                    container.close()
                except Exception:
                    pass

            # If still running, we will loop and try to reconnect
            if self._running:
                print("[Worker] Will attempt to reconnect in 1s")
                time.sleep(1.0)

        self.finished.emit()
        print("[Worker] Stopped")

    def stop(self):
        self._running = False


class VideoWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.setMinimumSize(640, 480)

    @QtCore.Slot(np.ndarray)
    def update_frame(self, frame: np.ndarray):
        # frame is BGR uint8
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(
            frame.data, w, h, bytes_per_line, QtGui.QImage.Format_BGR888
        ).copy()  # copy to detach from numpy buffer
        self._pixmap = QtGui.QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.black)

        if self._pixmap is not None:
            # scale to fit while keeping aspect ratio
            target = self._pixmap.scaled(
                self.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            x = (self.width() - target.width()) // 2
            y = (self.height() - target.height()) // 2
            painter.drawPixmap(x, y, target)

        painter.end()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Gaze Prototype")

        # Central video widget
        self.video_widget = VideoWidget(self)
        self.setCentralWidget(self.video_widget)

        # Simple toolbar with an Exit button for pre-MVP
        exit_action = QtGui.QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        toolbar = self.addToolBar("Main")
        toolbar.addAction(exit_action)

        # Status bar for basic info / errors
        self.status = self.statusBar()
        self.status.showMessage("Connecting to camera...")

        # Thread + worker setup
        self.thread = QtCore.QThread(self)
        self.worker = VideoWorker()
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.start)
        self.worker.frameReady.connect(self.video_widget.update_frame)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @QtCore.Slot(str)
    def on_worker_error(self, msg: str):
        self.status.showMessage(msg)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Stop worker and wait for thread to finish
        if self.worker is not None:
            self.worker.stop()
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(2000)
        return super().closeEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(960, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
