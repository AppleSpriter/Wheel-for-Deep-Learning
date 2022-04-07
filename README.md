# Wheel-for-Deep-Learning
Some image processing, data processing scripts, used to assist in-depth learning training

# Usage/用法

## windows-imagename-processing/windows图像处理

主要针对windows中图像命名统一问题，使用步骤：

1. 使用``去除空格.bat`脚本去除当前文件夹下所有文件内的空格。
2. 使用`获取文件名.bat`脚本将当前文件夹下所有文件名获取并存储到1.xls文件中。
3. 使用 `重命名.xls`将要修改的文件名进行数字自增等处理，最后使用`=D2&" "&A2&" "&B2` 其中D2为windows 修改文件名的命令`ren`，A2为原文件名，B2为需要修改的文件名。
4. 复制`重命名.xls`最后一列中的修改文件名代码到`ANSI修改.bat`中，一旦有中文情况，如果格式为UTF-8请另存为修改为ANSI运行。