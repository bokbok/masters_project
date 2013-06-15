#!/usr/bin/ruby
snip_point = false
first = true
File.open(ARGV[0]) do |file|
  File.open(ARGV[0] + '.snipped', 'w') do |out|
     file.each do |line|
        if /^t=#{ARGV[1]}$/ =~ line
	  snip_point = true
	  puts "Found snippoint"
	end
        
	if snip_point || first
	  out.puts(line)
	  first = false
	end
     end
  end
end
