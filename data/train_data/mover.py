import os
import shutil

# for f in os.listdir()
for f in os.listdir():
    if '.txt' in f and f[0].isalpha():
        folderPath = os.path.join(
            os.getcwd(),
            'old',
            f[0].upper()
        )
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
        filePath = os.path.join(os.getcwd(), f)
        shutil.move(filePath, folderPath)

