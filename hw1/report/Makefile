FILE	= hw1
TEX		= $(FILE).tex
XDV		= $(FILE).xdv

all: tex

tex: $(TEX)
	xelatex -no-pdf $(FILE)
	bibtex $(FILE)
	xelatex -no-pdf $(FILE)
	xelatex -no-pdf $(FILE)
	xdvipdfmx $(XDV)
	make clean

clean:
	rm -f *.aux *.dvi *.bbl *.blg *.idx *.ilg *.ind *.log *.nav *.out *.snm *.xdv *.toc
