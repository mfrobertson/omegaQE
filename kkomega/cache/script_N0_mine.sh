declare -a typs=("TT" "EE" "TE" "TB" "EB")
for typ in "${typs[@]}"
do
echo "$typ"
python N0_mine.py SO "$typ" False gradient 800
done

declare -a gmv_typs=("TE" "EB" "TEB")
for typ in "${gmv_typs[@]}"
do
echo "$typ"
python N0_mine.py SO "$typ" True gradient 400
done

