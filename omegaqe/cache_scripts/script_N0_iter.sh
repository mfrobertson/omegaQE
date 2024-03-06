#declare -a exps=("SO_base" "SO_goal" "S4_base")
declare -a exps=("SO_goal" "S4_base" "S4_dp")
for exp in "${exps[@]}"
do
echo "$exp"
python N0_iter.py "$exp" EB True 15 30 3000 30 5000 False
python N0_iter.py "$exp" EB True 15 30 3000 30 5000 True
done

