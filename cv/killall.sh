screen -ls  | grep ak | cut -d. -f1 | awk '{print $1}' | xargs kill

