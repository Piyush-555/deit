MISSING=0

for f in /mnt/proj3/open-35-39/datasets/coyo300m/*.tar; do
  base=$(basename "$f" .tar)
  if [ ! -f "/mnt/proj3/open-35-39/datasets/coyo300m/${base}_stats.json" ]; then
    echo "Missing stats for: $base"
    echo "Deleting /mnt/proj3/open-35-39/datasets/coyo300m/${base}.tar"
    rm "/mnt/proj3/open-35-39/datasets/coyo300m/${base}.tar"
    MISSING=$((MISSING + 1))
  fi
done

echo "Total shards without stats: $MISSING"
