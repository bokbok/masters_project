1.upto(500) do |num|
   param_set = sprintf("%05d.ode", num)

   puts `python python/burst_topology_thal.py #{ARGV[0]} #{param_set} #{ARGV[1]} #{ARGV[2]}`
end
