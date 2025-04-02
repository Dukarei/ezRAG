from pypdf import PdfMerger
# make list of pdf names
pdfs = [f"GLSL{x}.pdf" for x in range(347)]

#merge to master file
merger = PdfMerger()

for pdf in pdfs:
    merger.append(pdf)

merger.write("GLSLmaster.pdf")
merger.close()

#work on the below to merge text files as well
import os

dir_path = '/path/to/your/directory'

for filename in os.listdir(dir_path):
    file_path = os.path.join(dir_path, filename)
    if os.path.isfile(file_path):
        print(file_path)