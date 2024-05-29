#ifndef _GOGAME
#define _GOGAME

#include <iostream>
#include <cstdlib>
#include <vector>
#include <type_traits>

const int MAX_N = 21;

template <typename T>
struct coord{
    T x, y;
    coord(): x(0), y(0){}
    coord(T x, T y): x(x), y(y){}

    coord operator+(const coord &b) const {
        return coord(x + b.x, y + b.y);
    }

    bool operator==(const coord<T> &b) const {
        return x == b.x && y == b.y;
    }
};

inline std::ostream &operator<<(std::ostream &os, const coord<int> &pos) {
    os << "(" << pos.x << ", " << pos.y << ")";
    return os;
}

template <typename T>
struct array2d{
private:
    static_assert(!std::is_same_v<T, bool>, "f**k vector<bool>");
    std::vector<T> data;
    int n, m;
public:
    array2d(int n, int m) : data(n * m), n(n), m(m) {}
    array2d(int n, int m, const T &default_value)
        : data(n * m, default_value), n(n), m(m) {}

    int get_n() const { return n; }
    int get_m() const { return m; }

    T* operator[](int i) { return data.data() + i * m; }
    const T* operator[](int i) const { return data.data() + i * m; }

    T &operator[](const coord<int> &pos) { return data[pos.x * m + pos.y]; }
    const T &operator[](const coord<int> &pos) const { return data[pos.x * m + pos.y]; }

    bool is_valid_pos(const coord<int> &pos) const {
        return pos.x >= 0 && pos.x < n && pos.y >= 0 && pos.y < m;
    }

    const std::vector<T> & get_data()const{
        return data;
    }

    T* get_data_ptr(){
        return data.data();
    }
};

class c_GoBoard {
public:
    int n;

private:

    array2d<int> board;
    
    coord<int> ko_pos;
    
    int consecutive_pass;
    
    int step_cnt;

    coord<int> last_step_pos;
    int last_step_color;

    int count_stone(const int &color);

public:
    c_GoBoard(const int &n);

    bool add_stone(const coord<int> &pos, const int &color); // add a stone at pos, return false if illegal
    void pass_stone(const int &color); // take an action that no stone is added
    const std::vector <int> & get_board() const; // get the current board as an array
    void set_board(const std::vector <int> &board_data); // set the current board by an array
    const std::vector <coord<int> > get_legal_moves(const int &color); // get all legal moves for a specific color
    int is_game_over(); // check if the game is over, and return the winner if it is
    coord<float> get_score(); // get the score of the game
};

#endif