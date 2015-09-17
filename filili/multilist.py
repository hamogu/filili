def read_multipletlist(filename):
  multipletlist=[]
  with open(filename, 'r') as linefile:
    newmultiplet={'name':'','wave':[],'linename':[]}
    for read in linefile:
      #cut off all comments on line
      read=read.split('#')[0]
      #strip whitespaces
      read=read.strip()
      #skip empty lines or comment lines
      if not(len(read) == 0):
        readsplit = read.split('&')
        if len(readsplit) == 1:
          if len(newmultiplet['wave']) > 0:
            multipletlist.append(newmultiplet)
          newmultiplet={'name':readsplit[0],'wave':[],'linename':[]}
        else:
          #add wavelength and linename to newmultiplet
          try:
            wave=float(readsplit[0])
            if wave <= 0: raise ValueError
            newmultiplet['wave'].append(wave)
            newmultiplet['linename'].append(readsplit[1].strip())
          except ValueError:
            print readsplit[0]+' is not a valid number in line '+read
    else:
       #at end of file (i.e. end of for loop) add last multiplet (if not empty)
       if len(newmultiplet['wave']) > 0:
         multipletlist.append(newmultiplet)
  return multipletlist
