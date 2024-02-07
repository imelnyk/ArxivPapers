from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

class GDrive():
    def __init__(self, gid):
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        self.drive = GoogleDrive(gauth)
        self.gid = gid

    def reconnect(self):
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        self.drive = GoogleDrive(gauth)

    def upload_audio(self, google_filename, local_filename):
        file_metadata = {'parents': [{'id': self.gid}], 'title': google_filename}
        media = self.drive.CreateFile(file_metadata)
        media.SetContentFile(local_filename)
        media.Upload()


