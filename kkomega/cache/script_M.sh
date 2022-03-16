declare -a arr=("kappa-kappa","gal-gal" "gal-kappa" "kappa-gal")
for typ in "${arr[@]}"
do
echo "$typ"
python M.py 4000 1000 "True" "$typ"
done
