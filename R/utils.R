


progress = function(i,total,steps=10) {
  perc = round(100*i/total)
  sofar = floor(steps * i/total)
  paste0(
    sprintf('%3s',perc),
    "% [",
    paste0(rep('=',sofar),collapse=''),
    paste0(rep(' ',steps-sofar),collapse=''),
    "]"
  )
}

time_since = function(t) format(as.POSIXct(as.numeric(Sys.time()) - t, tz="UTC"), '%H:%M:%S')

