# 📋 Submission Instruction

Here we provide sample `prediction.zip` files for both 2D and 3D predictions (Note: The actual prediction files are just
for demonstration purposes and do not contain any meaningful data). We also include platform-specific instructions on
how to package your predictions 🗂️.

## 🖥️ Windows

Since we don't want to include the parent folder when you zip the prediction files, you can follow these steps:

- Open Windows File Explorer and navigate to the folder you want to zip.

- `Ctrl + A` selects the prediction json/nii.gz files you want to zip, but do not select the parent folder.

- Right-click on the selected folder and choose "Send to" -> "Compressed (zipped) folder".

- In the pop-up window, name the zip file as `predict.zip`.

- Wait for the compression to complete, and you will find the `predict.zip` file in the same directory.

## 🐧 Ubuntu

We only want to zip all the prediction files inside foldername. Thus, you can use the following command:

```
cd foldername/
zip -r ../predict.zip .
```

Explanation:

- The `cd foldername/` command navigates into the foldername directory.
- The `zip -r ../predict.zip .` command:
  - The `-r` option enables recursive compression of subdirectories.
  - `../predict.zip` specifies that the compressed output should be saved as `predict.zip` in the parent directory.
  - The `.` represents all the files and subdirectories in the current directory.

In this way, only the contents of the foldername directory will be added to the `predict.zip` file, without including the
foldername directory itself.
