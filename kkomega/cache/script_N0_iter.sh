declare -a exps=("SO_base" "SO_goal" "S4_base")
for exp in "${exps[@]}"
do
echo "$exp"
python N0_iter.py "$exp" TEB True 3 30 3000 30 5000 False
python N0_iter.py "$exp" TEB True 3 30 3000 30 5000 True
done

