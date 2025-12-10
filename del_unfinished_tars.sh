cd /mnt/proj3/open-35-39/datasets/coyo300m/

# Delete .tar shards 190 and above
for i in $(seq -w 24897 30000); do
    [ -f "${i}.tar" ] && rm "${i}.tar"
done

# Delete .parquet shards 190 and above
for i in $(seq -w 24897 30000); do
    [ -f "${i}.parquet" ] && rm "${i}.parquet"
done

# Delete .json shards 190 and above
for i in $(seq -w 24897 30000); do
    [ -f "${i}_stats.json" ] && rm "${i}_stats.json"
done

rm -r _tmp