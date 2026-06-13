


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

time_since = function(t) strftime(as.numeric(Sys.time()) - t - 57600, '%H:%M:%S') #why does strftime(0, '%H') == 16??

