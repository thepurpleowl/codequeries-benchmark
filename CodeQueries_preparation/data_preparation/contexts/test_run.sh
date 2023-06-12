#!/bin/sh
# echo $(ls|grep test_)
for python_file_name in $(ls|grep test_)
do
   python $python_file_name
done