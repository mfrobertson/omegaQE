#declare -a typs=("TT" "EE" "TE" "TB" "EB")
declare -a typs=("TT")
for typ in "${typs[@]}"
do
echo "$typ"
python N0_mine.py SO_base "$typ" False gradient 1000 30 3000 30 5000
python N0_mine.py SO_goal "$typ" False gradient 1000 30 3000 30 5000
python N0_mine.py S4_base "$typ" False gradient 1000 30 3000 30 5000
done

#declare -a gmv_typs=("TE" "EB" "TEB")
declare -a gmv_typs=("EB" "TEB")
for typ in "${gmv_typs[@]}"
do
echo "$typ"
python N0_mine.py SO_base "$typ" True gradient 500 30 3000 30 5000
python N0_mine.py SO_goal "$typ" True gradient 500 30 3000 30 5000
python N0_mine.py S4_base "$typ" True gradient 500 30 3000 30 5000
done

