#include <vector>

namespace keymolen {

	class HoughCircle {
	public:
		HoughCircle();
		virtual ~HoughCircle();
	public:
		int Transform(unsigned char* img_data, int w, int h, int r);
		int GetCircles(int threshold, std::vector< std::pair< std::pair<int, int>, int> >& result );
	    const unsigned int* GetAccu(int *w, int *h);
	private:
		unsigned int* _accu;
		int _accu_w;
		int _accu_h;
		int _img_w;
		int _img_h;
		int _r;

	};

}