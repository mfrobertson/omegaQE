declare -a arr=("my_SA" "my_SPT" "my_S4" "my_S5" "my_SO")
for exp in "${arr[@]}"
do
echo "$exp"
python N0.py "$exp" 14 14 TQU
done
