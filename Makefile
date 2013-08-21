all:
	make -C layers/cpp/
	make -C cuda/
test:
	nosetests -v tests/*.py
