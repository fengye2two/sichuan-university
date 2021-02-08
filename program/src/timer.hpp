#ifndef _TIMER_HPP_
#define _TIMER_HPP_

#include <chrono>

namespace oak
{
	class Timer {
	public:
		Timer() : start_(), end_() {
		}

		void Start() {
			start_ = std::chrono::system_clock::now();
		}

		void Stop() {
			end_ = std::chrono::system_clock::now();
		}

		double GetElapsedMilliseconds() {
			return (double)std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
		}

	private:
		std::chrono::time_point<std::chrono::system_clock> start_;
		std::chrono::time_point<std::chrono::system_clock> end_;
	};
}

#endif // _TIMER_HPP_