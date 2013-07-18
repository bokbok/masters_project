#!/usr/bin/ruby
first = true
File.open(ARGV[0]) do |file|
  File.open(ARGV[0] + ".extract_#{ARGV[1]}_#{ARGV[2]}", 'w') do |out|
     time = nil
     file.each do |line|
	if first
	   out.puts("t " + line)	 
	   first = false
	elsif line =~ /^t=\d+/ 
	   time = line.gsub("t=", "").gsub("\n", "")
	elsif line =~ /\(#{ARGV[1]},#{ARGV[2]}\):/
           puts "Extract for t=#{time}"
	   out.puts(time + " " + line.gsub(/\(#{ARGV[1]},#{ARGV[2]}\):/, ""))
        end
     end
  end
end
