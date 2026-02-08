#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <fstream>
#include <unordered_set>
using namespace std;



// ============================================================
// ЭТАП 1.
// ============================================================

// -------------------- RandomStreamGen ------------------------
class RandomStreamGen {
public:
    RandomStreamGen(size_t n, uint64_t seed)
        : n_(n), rng_(seed) {}

    vector<string> generateStream() {
        vector<string> s;
        s.reserve(n_);
        for (size_t i = 0; i < n_; ++i) s.push_back(genOne());
        return s;
    }

    vector<size_t> splitPrefixesByPercent(int stepPercent) const {
        if (stepPercent <= 0 || stepPercent > 100) {
            throw invalid_argument("stepPercent must be in [1..100]");
        }
        vector<size_t> pref;
        for (int p = stepPercent; p <= 100; p += stepPercent) {
            size_t k = (size_t)((__int128)n_ * p / 100);
            if (k == 0) k = 1;
            if (k > n_) k = n_;
            if (pref.empty() || pref.back() != k) pref.push_back(k);
        }
        if (pref.empty() || pref.back() != n_) pref.push_back(n_);
        return pref;
    }

    size_t size() const { return n_; }

private:
    size_t n_;
    mt19937_64 rng_;

    static constexpr const char* ALPHABET =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-";

    char randChar() {
        static constexpr int ALPH_LEN = 26 + 26 + 10 + 1;
        uniform_int_distribution<int> d(0, ALPH_LEN - 1);
        return ALPHABET[d(rng_)];
    }

    string genOne() {
        uniform_int_distribution<int> lenDist(1, 30);
        int L = lenDist(rng_);
        string x;
        x.resize(L);
        for (int i = 0; i < L; ++i) x[i] = randChar();
        return x;
    }
};

// -------------------- HashFuncGen ----------------------------

static inline uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}
static inline uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

static uint32_t murmur3_32(const uint8_t* data, size_t len, uint32_t seed) {
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    uint32_t h1 = seed;
    const int nblocks = (int)(len / 4);


    const uint32_t* blocks = (const uint32_t*)(data);
    for (int i = 0; i < nblocks; i++) {
        uint32_t k1 = blocks[i];
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }


    const uint8_t* tail = data + nblocks * 4;
    uint32_t k1 = 0;
    switch (len & 3) {
        case 3: k1 ^= (uint32_t)tail[2] << 16;
        case 2: k1 ^= (uint32_t)tail[1] << 8;
        case 1: k1 ^= (uint32_t)tail[0];
                k1 *= c1;
                k1 = rotl32(k1, 15);
                k1 *= c2;
                h1 ^= k1;
    }


    h1 ^= (uint32_t)len;
    h1 = fmix32(h1);
    return h1;
}

class HashFuncGen {
public:
    explicit HashFuncGen(uint32_t seed = 0x9747b28cU) : seed_(seed) {}

    uint32_t operator()(const string& s) const {
        return murmur3_32(reinterpret_cast<const uint8_t*>(s.data()), s.size(), seed_);
    }

private:
    uint32_t seed_;
};


// -------------------- UniformityReport ----------------------------
struct UniformityReport {
    size_t n;           // количество элементов
    size_t buckets;     // количество корзин
    size_t minCount;    // минимальное количество в корзине
    size_t maxCount;    // максимальное количество в корзине
    double mean;        // среднее количество на корзину
    double stddev;      // стандартное отклонение
    double coeffVar;    // коэффициент вариации (stddev/mean)
    double chiSquare;   // статистика хи-квадрат
};

// Функция для тестирования равномерности распределения хеш-функции
template<typename HashFunc>
UniformityReport testUniformity(const vector<string>& stream, const HashFunc& hf, size_t numBuckets) {
    vector<size_t> bucketCounts(numBuckets, 0);
    
    // Распределяем элементы по корзинам
    for (const auto& s : stream) {
        uint32_t h = hf(s);
        size_t bucket = h % numBuckets;
        bucketCounts[bucket]++;
    }
    
    // Вычисляем статистики
    size_t minCount = *min_element(bucketCounts.begin(), bucketCounts.end());
    size_t maxCount = *max_element(bucketCounts.begin(), bucketCounts.end());
    
    double mean = (double)stream.size() / numBuckets;
    
    // Стандартное отклонение
    double variance = 0.0;
    for (size_t count : bucketCounts) {
        double diff = count - mean;
        variance += diff * diff;
    }
    variance /= numBuckets;
    double stddev = sqrt(variance);
    
    double coeffVar = (mean > 0) ? (stddev / mean) : 0.0;
    
    // Хи-квадрат статистика
    double chiSquare = 0.0;
    for (size_t count : bucketCounts) {
        double diff = count - mean;
        chiSquare += (diff * diff) / mean;
    }
    
    return UniformityReport{
        stream.size(),
        numBuckets,
        minCount,
        maxCount,
        mean,
        stddev,
        coeffVar,
        chiSquare
    };
}

// -------------------- Demo main --------------
// Небольшая демонстрация работы Этапа 1.
int main() {
    size_t N = 100000;
    uint64_t seedStream = 42;

    RandomStreamGen gen(N, seedStream);
    auto stream = gen.generateStream();


    auto pref = gen.splitPrefixesByPercent(10);
    cout << "Prefixes (10% steps): ";
    for (auto x : pref) cout << x << " ";
    cout << "\n";

    HashFuncGen hf(123456789u);
    auto rep = testUniformity(stream, hf, 2048);

    cout << "Uniformity test:\n";
    cout << " n=" << rep.n << ", buckets=" << rep.buckets << "\n";
    cout << " min=" << rep.minCount << ", max=" << rep.maxCount << "\n";
    cout << " mean=" << rep.mean << ", stddev=" << rep.stddev
         << ", CV=" << rep.coeffVar << "\n";
    cout << " chiSquare=" << rep.chiSquare << " (df" << (rep.buckets - 1) << ")\n";

    return 0;
}
