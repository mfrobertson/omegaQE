EXP=$1
declare -a gmv_typs=("EB" "TEB")
for typ in "${gmv_typs[@]}"
do
echo "$typ"
python N0_iter.py $EXP "$typ" True 7 30 3000 30 5000 False
python N0_iter.py $EXP "$typ" True 7 30 3000 30 5000 True
done

declare -a gmv_typs=("TT")
for typ in "${gmv_typs[@]}"
do
echo "$typ"
python N0_iter.py $EXP "$typ" False 7 30 3000 30 5000 False
python N0_iter.py $EXP "$typ" False 7 30 3000 30 5000 True
done

python N0_plancklens.py $EXP  30 3000 30 5000

#declare -a typs=("TT" "EE" "TE" "TB" "EB")
declare -a typs=("TT")
for typ in "${typs[@]}"
do
echo "$typ"
#python N0_plancklens.py $EXP "$typ" False gradient 1000 30 3000 30 5000
done

declare -a gmv_typs=("EB" "TEB")
#declare -a gmv_typs=("EB")
for typ in "${gmv_typs[@]}"
do
echo "$typ"
#python N0_plancklens.py $EXP "$typ" True gradient 500 30 3000 30 5000
done

