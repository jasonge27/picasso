#-----------------------------------------------------------------------------------#
# Package: picasso                                                                  #
# scio.plot(): graph visualization                                                  #
# Author: Jian Ge, Xingguo Li                                                       #
# Email: <jiange@princeton.edu>, <xingguo.leo@gmail.com>                            #
# Date: Sep 1st, 2016                                                               #
# Version: 0.5.1                                                                    #
#-----------------------------------------------------------------------------------#

scio.plot <- function(G, epsflag = FALSE, graph.name = "default", cur.num = 1, location=NULL){
  gcinfo(FALSE)
  if(missing(location))	location = getwd()
  setwd(location)
  g = graph.adjacency(as.matrix(G!=0), mode="undirected", diag=FALSE)
  layout.grid = layout.fruchterman.reingold(g)
  
  if(epsflag == TRUE)	postscript(paste(paste(graph.name, cur.num, sep=""), "eps", sep="."), width = 8.0, height = 8.0)             
  par(mfrow = c(1,1))
  plot(g, layout=layout.grid, edge.color='gray50',vertex.color="red", vertex.size=2, vertex.label=NA)
  rm(g,location)	
  gc()
  if(epsflag == TRUE) dev.off()
}
