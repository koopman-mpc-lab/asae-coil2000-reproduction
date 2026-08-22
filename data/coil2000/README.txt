Official UCI COIL 2000 extracts used by src.data.coil_loader:

  ticdata2000.txt
  ticeval2000.txt
  tictgts2000.txt
  TicDataDescr.txt

Source: https://archive.ics.uci.edu/dataset/125/insurance+company+benchmark+coil+2000
Mirror: http://kdd.ics.uci.edu/databases/tic/

The loader joins ticeval2000.txt with tictgts2000.txt on row order and
concatenates the 5822 labelled training rows with the 4000 evaluation rows
before the archived stratified 70/15/15 split (411 / 88 / 87 positives).
