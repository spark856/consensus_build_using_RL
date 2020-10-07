
#!/bin/bash

array=(
"3 3 2"
"3 2.5 2"
"3 2 2.5"
)

for var in "${array[@]}"
do
    nohup python3 play/wtest.py ${var} &
done
