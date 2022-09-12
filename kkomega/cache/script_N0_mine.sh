declare -a typs=("TT" "EE" "TE" "TB" "EB")
for typ in "${typs[@]}"
do
echo "$typ"
python N0_mine.py S4_base "$typ" False gradient 1000 30 3000 30 5000
done

declare -a gmv_typs=("TE" "EB" "TEB")
for typ in "${gmv_typs[@]}"
do
echo "$typ"
python N0_mine.py S4_base "$typ" True gradient 500 30 3000 30 5000
done

