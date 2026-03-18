allowed_formats = ['mp4', 'avi', 'mov', 'mkv']

def allowed_file(filename):
    return '.' in filename and filename.rsplit(".", 1)[1].lower() in allowed_formats