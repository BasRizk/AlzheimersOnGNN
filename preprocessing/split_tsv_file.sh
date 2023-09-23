!/bin/bash

# Check for the correct number of arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 input_file n [output_dir]"
    echo "Split a TSV file into n smaller files"
    echo "If output_dir is not specified, the files will be created in the current directory"
    exit 1
fi

input_file="$1"   # The input TSV file
n="$2"            # Number of files to split into
output_dir="${3:-.}" # output_directory or default to current directory

base_name=$(basename "$input_file")
output_prefix="${output_dir}/${base_name%.*}" # Prefix for the output files (removing the extension)

echo "Splitting $input_file into $n files with $output_prefix as prefix"

# Calculate the number of lines in each output file (excluding the header)
total_lines=$(wc -l < "$input_file")
lines_per_file=$(( (total_lines - 1) / n ))  # Subtract 1 to exclude the header


echo "Total lines: $total_lines"
echo "Lines per file: $lines_per_file"

FILE_HEADER=$(head -n 1 "$input_file")

echo "File header: $FILE_HEADER"

# Define the split_filter function to append FILE_HEADER on top of content of FILE

# split_filter() { {head -n 1 $input_file; cat;} > $FILE }
# export -f split_filter


# echo "Setting up split_filter function"
# split_filter () { { head -n 1 $input_file; cat; } > "$FILE"; }; 
# export -f split_filter; 

# tail -n +2 $input_file | split --lines=$lines_per_file  --filter=split_filter - ${output_prefix}_ 

trap 'rm  ${output_prefix}_*  tmp_file ; exit 13' SIGINT SIGTERM SIGQUIT 
tail -n +2 $input_file | split -d -l $lines_per_file - ${output_prefix}_
for file in ${output_prefix}_*;
do
    head -n 1 $input_file > tmp_file
    cat $file >> tmp_file
    mv -f tmp_file $file
done

echo "Splitting $input_file into $n files with $lines_per_file lines each"

echo "Split into $n files: ${output_prefix}_*"
