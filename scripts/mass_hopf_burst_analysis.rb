1.upto(500) do |num|
   param_set = sprintf("%05d.ode", num)

   puts `python python/burst_hopf_topology.py #{ARGV[0]} #{param_set}`
end
