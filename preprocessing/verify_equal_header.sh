#!/bin/bash

# Check if at least two arguments are provided
if [ $# -lt 2 ]; then
  echo "Usage: $0 <file1> <file2> [<file3> ...]"
  exit 1
fi

# Get the first file from the command-line arguments
first_file="$1"
shift

# Get the first line of the first file
first_line=$(head -n 1 "$first_file")

# Loop through each remaining file
for file in "$@"; do
  # Get the first line of the current file
  current_first_line=$(head -n 1 "$file")

  # If the first line of this file is different from the first line of the first file, print the difference
  if [ "$first_line" != "$current_first_line" ]; then
    echo "Difference found in $file:"
    echo "First Line in $first_file: $first_line"
    echo "First Line in $file: $current_first_line"
    exit 1
  fi
done

exit 0