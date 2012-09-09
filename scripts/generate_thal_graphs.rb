Dir["parameterisations/**/*.ode-burst-thal.yml"].each do |f|
   puts `python python/display_thal.py #{f} save`
end
