#include "board.h"

#include <cstring>
#include <queue>

const int INF = 1e9;
const int WHITE = -1, BLACK = 1, EMPTY = 0, DRAW = 2;

const int dx[4] = {0, 0, 1, -1};
const int dy[4] = {1, -1, 0, 0};


c_GoBoard::c_GoBoard(const int &n):
    n(n),
    board(n, n),
    ko_pos(-INF, -INF),
    consecutive_pass(0),
    step_cnt(0),
    last_step_pos(-INF, -INF),
    last_step_color(-INF)
{
    if (n > MAX_N){
        std::cerr << "Board size(" << n << ") too large! (max:" << MAX_N << ") " << std::endl;
        throw std::exception();
    }
}

const std::vector <int> & c_GoBoard::get_board() const{
    return board.get_data();
}

void c_GoBoard::set_board(const std::vector <int> &board_data){
    if (board_data.size() != (unsigned int) n * n){
        std::cerr << "[c_GoBoard::set_board] invalid board data size: " << board_data.size() << " (expect: " << n * n << ")" << std::endl;
        throw std::exception();
    }
    std::memcpy(board.get_data_ptr(), board_data.data(), sizeof(int) * n * n);
}

int c_GoBoard::count_stone(const int &color){
    int cnt = 0;
    for(int i=0; i<this->n; i++)
        for(int j=0; j<this->n; j++)
            if(this->board[i][j] == color)
                cnt++;
    return cnt;
}

void c_GoBoard::pass_stone(const int &color){
    step_cnt += 1;
    consecutive_pass += 1;
    ko_pos = coord<int>(-INF, -INF);
}


bool c_GoBoard::add_stone(const coord<int> &pos, const int &color){
    // Add a stone at pos, throw exception if illegal
    // @param pos: position to add stone
    // @param color: color of the stone
    // @return: always true 
    // NOTE: this function not only add a stone, but also remove dead stones
    const int &x = pos.x, &y = pos.y;
    if (x < 0 || x >= n || y < 0 || y >= n){
        std::cerr << "[c_GoBoard::add_stone] invalid pos (" << x << ", " << y << ")" << std::endl;
        throw std::exception();
    }
    if (board[x][y] != EMPTY){
        std::cerr << "[c_GoBoard::add_stone] position not empty: pos=(" << x << ", " << y << "), color here=" << board[x][y] << std::endl;
         throw std::exception();
    }
    board[x][y] = color;
    // ======================
    // TODO: Your code here
    // ======================
    consecutive_pass = 0;
    step_cnt += 1;
    last_step_pos = pos;
    last_step_color = color;
    return true;
}

const std::vector <coord<int> > c_GoBoard::get_legal_moves(const int &color){
    // Get all legal moves for a specific color
    // @param color: color of the stones
    // @return: a list of legal moves
    std::vector <coord<int> > legal_moves;
    // ======================
    // TODO: Your code here
    // ======================
    return legal_moves;
}

coord<float> c_GoBoard::get_score(){
    // Get the score of the game
    // @return: a pair of float, (black score, white score)
    // NOTE: the score is the number of stones and empty points captured by each player
    float scores[2] = {.0, .0};// black, white
    // ======================
    // TODO: Your code here
    // ======================
    return coord<float>(scores[0], scores[1]); // black, white
}

int c_GoBoard::is_game_over(){
    // Check if the game is over, and return the winner if it is
    // @return: the winner of the game (BLACK, WHITE or DRAW), or EMPTY if the game is not over
    if (step_cnt >= n * n * 2 + 1 || consecutive_pass >= 2){
        auto scores = get_score();
        if (scores.x > scores.y){
            return BLACK;
        }
        else if (scores.x < scores.y){
            return WHITE;
        }
        else{
            return DRAW;
        }
    }
    return EMPTY;
}