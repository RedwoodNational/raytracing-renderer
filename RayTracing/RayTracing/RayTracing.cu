#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <utility>
#include <array>
#include <set>
#include <cuda.h>
#include <omp.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err)
            << " in " << std::string(file)
            << " at line " << line << "\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

struct vector3d_t {
    __device__ __host__
    friend vector3d_t operator*(double, const vector3d_t);

    __device__ __host__
    bool operator < (vector3d_t const rhs) const {
        if (x < rhs.x) {
            return true;
        }
        if (x > rhs.x) {
            return false;
        }
        if (y < rhs.y) {
            return true;
        }
        if (y > rhs.y) {
            return false;
        }
        if (z < rhs.z) {
            return true;
        }
        if (z > rhs.z) {
            return false;
        }
        return false;
    }

    __device__ __host__
    bool operator !=(vector3d_t v) const {
        if (std::abs(x - v.x) > 1e-6) {
            return true;
        }
        if (std::abs(y - v.y) > 1e-6) {
            return true;
        }
        if (std::abs(z - v.z) > 1e-6) {
            return true;
        }
        return false;
    }

    __device__ __host__
    static const vector3d_t k() {
        return vector3d_t{ 0, 0, 1 };
    }

    __device__ __host__
    static const vector3d_t j() {
        return vector3d_t{ 0, 1, 0 };
    }

    __device__ __host__
    static const vector3d_t i() {
        return vector3d_t{ 1, 0, 0 };
    }
    double x;
    double y;
    double z;

    __device__ __host__
    double norm() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    __device__ __host__
    vector3d_t get_unit() const {
        vector3d_t r(*this);
        return r /= norm();
    }

    __device__ __host__
    vector3d_t& make_unit() {
        operator /=(norm());
        return *this;
    }

    __device__ __host__
    vector3d_t reflect(const vector3d_t& n) const {
        return  2.0 * n * (n * (*this)) - (*this);
    }

    __device__ __host__
    vector3d_t transform(vector3d_t a, vector3d_t b, vector3d_t c) {
        vector3d_t r;
        r.x = a.x * x + b.x * y + c.x * z;
        r.y = a.y * x + b.y * y + c.y * z;
        r.z = a.z * x + b.z * y + c.z * z;
        return r;
    }

    __device__ __host__
    double operator* (vector3d_t const & v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __device__ __host__
    vector3d_t operator^(vector3d_t const & v) const  {
        vector3d_t r;
        r.x = y * v.z - z * v.y;
        r.y = z * v.x - x * v.z;
        r.z = x * v.y - y * v.x;
        return r;

    }

    __device__ __host__
    vector3d_t operator-(const vector3d_t& v) const {
        vector3d_t r(*this);
        r.x -= v.x;
        r.y -= v.y;
        r.z -= v.z;
        return r;
    }

    __device__ __host__
    vector3d_t operator+(const vector3d_t& v) const {
        vector3d_t r(*this);
        r.x += v.x;
        r.y += v.y;
        r.z += v.z;
        return r;
    }

    __device__ __host__
    vector3d_t& operator *=(const double m) {
        x *= m;
        y *= m;
        z *= m;
        return *this;
    }

    __device__ __host__
    vector3d_t& operator /=(const double m) {
        x /= m;
        y /= m;
        z /= m;
        return *this;
    }

    __device__ __host__
    vector3d_t& operator +=(const vector3d_t& m) {
        x += m.x;
        y += m.y;
        z += m.z;
        return *this;
    }

    __device__ __host__
    vector3d_t operator /(const double m) const  {
        return vector3d_t(*this) /= m;
    }

    __device__ __host__
    vector3d_t operator *(const double m) const {
        return vector3d_t(*this) *= m;
    }

    __device__ __host__
    vector3d_t operator-() const {
        return vector3d_t{ -x, -y, -z };
    }
};

__device__ __host__
vector3d_t operator*(double a, const vector3d_t b) {
    return vector3d_t{ a * b.x, a * b.y, a * b.z };
}


struct triangle_t {
    vector3d_t v1;
    vector3d_t v2;
    vector3d_t v3;
    __device__ __host__
    vector3d_t normal () {
        return ((v2 - v1) ^ (v3 - v1)).get_unit();
    }
};

struct rectangle_t {
    vector3d_t v1;
    vector3d_t v2;
    vector3d_t v3;
    vector3d_t v4;
};

struct texture_t {
    int width;
    int height;
    uchar4* data;
};

struct intersection_t {
    bool hit;
    size_t i;
    double t;
};

struct real_color_t {
    double r;
    double g;
    double b;
    double a;
};

struct shape_t {
    real_color_t c;
    double ref;
    bool floor;
    triangle_t t;
    int floor_idx;
};

struct floor_t {
    rectangle_t rec;
    texture_t tex;
    real_color_t col;
};

enum class light_kind {
    ambient,
    point,
    diode,
};

struct light_t {
    light_kind t;
    vector3d_t p;
    real_color_t c;
};

struct context_t {
    vector3d_t nv;
    vector3d_t pv;
    vector3d_t dv;
    size_t ix;
};

struct tree_node_t {
    bool alive;
    size_t ix;
    real_color_t c;
    vector3d_t nv;
    vector3d_t pv;
    vector3d_t dv;
};

struct binding_t {
    size_t src_i;
    size_t dst_i;
    
};

struct payload_t {
    uchar4* pixels;
    binding_t* bindings;
    size_t size;
};



std::array<triangle_t, 36> dodecahedron() {
    const double p = (1 + std::sqrt(5)) / 2;
    const double u = 2 / (1 + std::sqrt(5));
    std::array<vector3d_t, 20> v = {
        vector3d_t{-u,  0,  p} / std::sqrt(3),
        vector3d_t{ u,  0,  p} / std::sqrt(3),
        vector3d_t{-1,  1,  1} / std::sqrt(3),
        vector3d_t{ 1,  1,  1} / std::sqrt(3),
        vector3d_t{ 1, -1,  1} / std::sqrt(3),
        vector3d_t{-1, -1,  1} / std::sqrt(3),
        vector3d_t{ 0, -p,  u} / std::sqrt(3),
        vector3d_t{ 0,  p,  u} / std::sqrt(3),
        vector3d_t{-p, -u,  0} / std::sqrt(3),
        vector3d_t{-p,  u,  0} / std::sqrt(3), 
        vector3d_t{ p,  u,  0} / std::sqrt(3),
        vector3d_t{ p, -u,  0} / std::sqrt(3),
        vector3d_t{ 0, -p, -u} / std::sqrt(3),
        vector3d_t{ 0,  p, -u} / std::sqrt(3),
        vector3d_t{ 1,  1, -1} / std::sqrt(3),
        vector3d_t{ 1, -1, -1} / std::sqrt(3),
        vector3d_t{-1, -1, -1} / std::sqrt(3),
        vector3d_t{-1,  1, -1} / std::sqrt(3),
        vector3d_t{ u,  0, -p} / std::sqrt(3),
        vector3d_t{-u,  0, -p} / std::sqrt(3)
    };
    std::array<triangle_t, 36> d;
    d[ 0] = triangle_t{ v[ 4], v[ 0], v[ 6] };
    d[ 1] = triangle_t{ v[ 0], v[ 5], v[ 6] };
    d[ 2] = triangle_t{ v[ 0], v[ 4], v[ 1] };
    d[ 3] = triangle_t{ v[ 0], v[ 3], v[ 7] };
    d[ 4] = triangle_t{ v[ 2], v[ 0], v[ 7] };
    d[ 5] = triangle_t{ v[ 0], v[ 1], v[ 3] };
    d[ 6] = triangle_t{ v[10], v[ 1], v[11] };
    d[ 7] = triangle_t{ v[ 3], v[ 1], v[10] };
    d[ 8] = triangle_t{ v[ 1], v[ 4], v[11] };
    d[ 9] = triangle_t{ v[ 5], v[ 0], v[ 8] };
    d[10] = triangle_t{ v[ 0], v[ 2], v[ 9] };
    d[11] = triangle_t{ v[ 8], v[ 0], v[ 9] };
    d[12] = triangle_t{ v[ 5], v[ 8], v[16] };
    d[13] = triangle_t{ v[ 6], v[ 5], v[12] };
    d[14] = triangle_t{ v[12], v[ 5], v[16] };
    d[15] = triangle_t{ v[ 4], v[12], v[15] };
    d[16] = triangle_t{ v[ 4], v[ 6], v[12] };
    d[17] = triangle_t{ v[11], v[ 4], v[15] };
    d[18] = triangle_t{ v[ 2], v[13], v[17] };
    d[19] = triangle_t{ v[ 2], v[ 7], v[13] };
    d[20] = triangle_t{ v[ 9], v[ 2], v[17] };
    d[21] = triangle_t{ v[13], v[ 3], v[14] };
    d[22] = triangle_t{ v[ 7], v[ 3], v[13] };
    d[23] = triangle_t{ v[ 3], v[10], v[14] };
    d[24] = triangle_t{ v[ 8], v[17], v[19] };
    d[25] = triangle_t{ v[16], v[ 8], v[19] };
    d[26] = triangle_t{ v[ 8], v[ 9], v[17] };
    d[27] = triangle_t{ v[14], v[11], v[18] };
    d[28] = triangle_t{ v[11], v[15], v[18] };
    d[29] = triangle_t{ v[10], v[11], v[14] };
    d[30] = triangle_t{ v[12], v[19], v[18] };
    d[31] = triangle_t{ v[15], v[12], v[18] };
    d[32] = triangle_t{ v[12], v[16], v[19] };
    d[33] = triangle_t{ v[19], v[13], v[18] };
    d[34] = triangle_t{ v[17], v[13], v[19] };
    d[35] = triangle_t{ v[13], v[14], v[18] };
    return d;
}


std::array<triangle_t, 20> icosahedron() {
    const double p = (1 + std::sqrt(5)) / 2;
    std::array<vector3d_t, 12> v = {
        vector3d_t{ 0, -1,  p} / std::sqrt(p + 2),
        vector3d_t{ 0,  1,  p} / std::sqrt(p + 2),
        vector3d_t{-p,  0,  1} / std::sqrt(p + 2),
        vector3d_t{ p,  0,  1} / std::sqrt(p + 2),
        vector3d_t{-1,  p,  0} / std::sqrt(p + 2),
        vector3d_t{ 1,  p,  0} / std::sqrt(p + 2),
        vector3d_t{ 1, -p,  0} / std::sqrt(p + 2),
        vector3d_t{-1, -p,  0} / std::sqrt(p + 2),
        vector3d_t{-p,  0, -1} / std::sqrt(p + 2),
        vector3d_t{ p,  0, -1} / std::sqrt(p + 2),
        vector3d_t{ 0, -1, -p} / std::sqrt(p + 2),
        vector3d_t{ 0,  1, -p} / std::sqrt(p + 2)
    };
    std::array<triangle_t, 20> i;
    i[ 0] = triangle_t{ v[ 0], v[ 1], v[ 2] };
    i[ 1] = triangle_t{ v[ 1], v[ 0], v[ 3] };
    i[ 2] = triangle_t{ v[ 0], v[ 2], v[ 7] };
    i[ 3] = triangle_t{ v[ 2], v[ 1], v[ 4] };
    i[ 4] = triangle_t{ v[ 4], v[ 1], v[ 5] };
    i[ 5] = triangle_t{ v[ 6], v[ 0], v[ 7] };
    i[ 6] = triangle_t{ v[ 3], v[ 0], v[ 6] };
    i[ 7] = triangle_t{ v[ 1], v[ 3], v[ 5] };
    i[ 8] = triangle_t{ v[ 4], v[ 5], v[11] };
    i[ 9] = triangle_t{ v[ 6], v[ 7], v[10] };
    i[10] = triangle_t{ v[ 3], v[ 6], v[ 9] };
    i[11] = triangle_t{ v[ 5], v[ 3], v[ 9] };
    i[12] = triangle_t{ v[ 7], v[ 2], v[ 8] };
    i[13] = triangle_t{ v[ 2], v[ 4], v[ 8] };
    i[14] = triangle_t{ v[ 9], v[10], v[11] };
    i[15] = triangle_t{ v[10], v[ 8], v[11] };
    i[16] = triangle_t{ v[ 5], v[ 9], v[11] };
    i[17] = triangle_t{ v[ 9], v[ 6], v[10] };
    i[18] = triangle_t{ v[ 7], v[ 8], v[10] };
    i[19] = triangle_t{ v[ 8], v[ 4], v[11] };

    return i;
}


std::array<triangle_t, 4> tetrahedron() {

    std::array<vector3d_t, 4> v{
        vector3d_t{ std::sqrt(8.0 / 9.0),  0,                      -1.0 / 3.0 },
        vector3d_t{-std::sqrt(2.0 / 9.0),  std::sqrt(2.0 / 3.0),   -1.0 / 3.0 },
        vector3d_t{-std::sqrt(2.0 / 9.0), -std::sqrt(2.0 / 3.0),   -1.0 / 3.0 },
        vector3d_t{0, 0, 1.0}
    };

    std::array<triangle_t, 4> t{
        triangle_t{v[1], v[0], v[2]},
        triangle_t{v[0], v[1], v[3]},
        triangle_t{v[2], v[0], v[3]},
        triangle_t{v[1], v[2], v[3]},
    };
    return t;
}



template<size_t s> void transform(std::array<triangle_t, s>& shapes,
    vector3d_t origin, double radius) {
    for (size_t i = 0; i < s; ++i) {
        shapes[i].v1 *= radius;
        shapes[i].v1 += origin;
        shapes[i].v2 *= radius;
        shapes[i].v2 += origin;
        shapes[i].v3 *= radius;
        shapes[i].v3 += origin;
    }

}

__device__ __host__
intersection_t intersect(   vector3d_t pv, vector3d_t dv, 
                            double t_min, double t_max, 
                            shape_t* scene, size_t n) {
    intersection_t r;
    r.hit = false;
    for (size_t i = 0; i < n; ++i) {
        triangle_t& s = scene[i].t;
        vector3d_t e1 = s.v2 - s.v1;
        vector3d_t e2 = s.v3 - s.v1;
        vector3d_t p = dv ^ e2;
        double norm = p * e1;
        if (std::abs(norm) < 1e-10) {
            continue;
        }
        vector3d_t t = pv - s.v1;
        double u = (p * t) / norm;
        if (u < 0.0 || u > 1.0) {
            continue;
        }
        vector3d_t q = t ^ e1;
        double v = (q * dv) / norm;
        if (v < 0.0 || v + u > 1.0) {
            continue;
        }
        double h = (q * e2) / norm;
        if (h < t_min || h > t_max) {
            continue;
        }
        if (!r.hit || h < r.t) {
            r.hit = true;
            r.t = h;
            r.i = i;
        }
    }

    return r;

}

__device__ __host__
double find_transparency(   vector3d_t pv, vector3d_t dv, 
                            double t_min, double t_max, 
                            shape_t* scene, size_t n) {
    double alpha = 1.0;
    for (size_t i = 0; i < n; ++i) {
        triangle_t& s = scene[i].t;
        vector3d_t e1 = s.v2 - s.v1;
        vector3d_t e2 = s.v3 - s.v1;
        vector3d_t p = dv ^ e2;
        double norm = p * e1;
        if (std::abs(norm) < 1e-10) {
            continue;
        }
        vector3d_t t = pv - s.v1;
        double u = (p * t) / norm;
        if (u < 0.0 || u > 1.0) {
            continue;
        }
        vector3d_t q = t ^ e1;
        double v = (q * dv) / norm;
        if (v < 0.0 || v + u > 1.0) {
            continue;
        }
        double h = (q * e2) / norm;
        if (h < t_min || h > t_max) {
            continue;
        }
        alpha *= scene[i].c.a;
    }

    return alpha;

}

__device__ __host__
real_color_t get_color( floor_t floor, shape_t* scene,
                        size_t ix, vector3d_t pv,
                        double min_x, double max_x, 
                        double min_y, double max_y) {
    real_color_t rc;
    if (!scene[ix].floor) {
        rc.r = scene[ix].c.r;
        rc.g = scene[ix].c.g;
        rc.b = scene[ix].c.b;
        rc.a = scene[ix].c.a;
        return rc;
    }
    vector3d_t bx = scene[scene[ix].floor_idx].t.v1 
        - scene[scene[ix].floor_idx].t.v2;
    vector3d_t by = scene[scene[ix].floor_idx].t.v3
        - scene[scene[ix].floor_idx].t.v2;
    vector3d_t bz = scene[scene[ix].floor_idx].t.normal();

    pv.transform(bx, by, bz);

    size_t x = static_cast<size_t>((pv.x - min_x)
                / (max_x - min_x) * (floor.tex.width - 1.0));
    size_t y = static_cast<size_t>((pv.y - min_y)
                / (max_y - min_y) * (floor.tex.height - 1.0));
    uchar4& pixel = floor.tex.data[y * floor.tex.width + x];
    rc.r = static_cast<double>(pixel.x / 255.0);
    rc.g = static_cast<double>(pixel.y / 255.0);
    rc.b = static_cast<double>(pixel.z / 255.0);
    rc.a = static_cast<double>(pixel.w / 255.0);
    return rc;
}

__device__ __host__
real_color_t compute_lighting(  vector3d_t pv, vector3d_t nv, 
                                vector3d_t vv, size_t hx,
                                shape_t* scene , size_t n,
                                light_t* lights, size_t m) {

    real_color_t x = { 0.0, 0.0, 0.0, 0.0 };
    for (size_t i = 0; i < m; ++i) {
        if (lights[i].t == light_kind::ambient) {
            x.r += lights[i].c.r;
            x.g += lights[i].c.g;
            x.b += lights[i].c.b;
        }

        if (lights[i].t == light_kind::diode) {
            vector3d_t dv = lights[i].p - pv;
            double r = 1.0 / dv.norm();
            x.r += lights[i].c.r * r * r;
            x.g += lights[i].c.g * r * r;
            x.b += lights[i].c.b * r * r;
        }


        if (lights[i].t == light_kind::point) {
            vector3d_t lv = lights[i].p - pv;
            double a = find_transparency(pv, lv, 1e-6, 1, scene, n);
            if (a < 1e-6) {
                continue;
            }
            
            double nv_dot_lv = nv * lv;
            if (nv_dot_lv > 0.0) {
                double c = a * nv_dot_lv / (nv.norm() * lv.norm());
                x.r += lights[i].c.r * c;
                x.g += lights[i].c.g * c;
                x.b += lights[i].c.b * c;
            }

            vector3d_t rv = lv.reflect(nv);
            double rv_dot_vv = rv * vv;
            if (rv_dot_vv > 0.0) {
                double c = a * rv_dot_vv / (rv.norm() * vv.norm());
                c = std::pow(c, scene[hx].ref);
                x.r += lights[i].c.r * c;
                x.g += lights[i].c.g * c;
                x.b += lights[i].c.b * c;
            }
        }
    }
    return x;
}


void cpu_render(payload_t payload,  size_t width, size_t height,
                shape_t* scene, size_t scene_size, floor_t floor,
                light_t* lights, size_t lights_size, size_t depth,
                double angle, vector3d_t cam, vector3d_t foc) {
    const double ratio =    static_cast<double>(height) /
                            static_cast<double>(width);
    const double dx = 2.0 / static_cast<double>(width  - 1);
    const double dy = 2.0 / static_cast<double>(height - 1);
    const double dz = 1.0 / std::tan(angle / 2.0);

    const vector3d_t bz = (foc - cam).make_unit();
    const vector3d_t bx = (bz ^ vector3d_t::k()).make_unit();
    const vector3d_t by = (bx ^ bz).make_unit();

    auto mmx = std::minmax({ floor.rec.v1.x, floor.rec.v2.x,
                             floor.rec.v3.x, floor.rec.v4.x });

    auto mmy = std::minmax({ floor.rec.v1.y, floor.rec.v2.y,
                             floor.rec.v3.y, floor.rec.v4.y });

    const double floor_min_x = mmx.first;
    const double floor_max_x = mmx.second;
    const double floor_min_y = mmy.first;
    const double floor_max_y = mmy.second;

    double floor_width  = mmx.second - mmx.first;
    double floor_height = mmy.second - mmy.first;

    const double inf = std::numeric_limits<double>::infinity();
    const double eps = 1e-6;

    const size_t tree_size = 1ull << depth;
    std::vector<tree_node_t> tree(tree_size);

#pragma omp parallel for firstprivate(tree)
    for(int u = 0; u < static_cast<int>(payload.size); ++u){

        size_t dst_i = payload.bindings[u].dst_i;
        size_t src_i = payload.bindings[u].src_i;
        size_t i = src_i % width, j = src_i / width;

        double r, g, b, a, w;
        intersection_t ret;
        vector3d_t pv, dv, nv;
        size_t ix;
        real_color_t s, c;
        size_t k, rc, lc;

        dv.x = (-1.0 + i * dx);
        dv.y = (+1.0 - j * dy) * ratio;
        dv.z = dz;
        dv = dv.transform(bx, by, bz);
        ret = intersect(cam, dv, 1, inf, scene, scene_size);
        tree[0].alive = ret.hit;

        if (tree[0].alive) {
            pv = cam + ret.t * dv;
            tree[0].dv = dv.make_unit();
            tree[0].pv = pv;
            tree[0].ix = ret.i;
            tree[0].nv = scene[ret.i].t.normal();
        }
        k = 0;
        while ((2 * k + 2) < tree_size) {
            lc = 2 * k + 1;
            rc = 2 * k + 2;
            if (!tree[k].alive) {
                tree[lc].alive = false;
                tree[rc].alive = false;
                ++k;
                continue;
            }
            dv = tree[k].dv;
            pv = tree[k].pv;
            nv = tree[k].nv;

            ret = intersect(pv, dv, eps, inf, scene, scene_size);
            tree[rc].alive = ret.hit;
            if (ret.hit) {
                tree[rc].ix = ret.i;
                tree[rc].nv = scene[ret.i].t.normal();
                tree[rc].pv = pv + ret.t * dv;
                tree[rc].dv = dv;
            }

            dv = (-dv).reflect(nv);

            ret = intersect(pv, dv, eps, inf, scene, scene_size);
            tree[lc].alive = ret.hit;
            if (tree[lc].alive) {
                tree[lc].ix = ret.i;
                tree[lc].nv = scene[ret.i].t.normal();
                tree[lc].pv = pv + ret.t * dv;
                tree[lc].dv = dv;
            }
            ++k;
        }

        while(k > 0){
            --k;
            lc = 2 * k + 1;
            rc = 2 * k + 2;
            if (!tree[k].alive) {
                tree[k].c.r = 0;
                tree[k].c.g = 0;
                tree[k].c.b = 0;
                continue;
            }


            dv = tree[k].dv;
            pv = tree[k].pv;
            nv = tree[k].nv;
            ix = tree[k].ix;
            c = get_color(floor, scene, ix, pv, 
                          floor_min_x, floor_max_x, 
                          floor_min_y, floor_max_y);
            s = compute_lighting(pv, nv, -dv, ix,
                                 scene, scene_size, 
                                 lights,lights_size);

            a = c.a;
            w = scene[ix].ref;


            r = std::min(1.0, c.r * s.r);
            g = std::min(1.0, c.g * s.g);
            b = std::min(1.0, c.b * s.b); 

            if (rc >= tree_size) {
                tree[k].c.r = r;
                tree[k].c.g = g;
                tree[k].c.b = b;
                continue;
            }
                
            c = tree[lc].c;
            r = std::min(1.0, (1.0 - w) * r + w * c.r);
            g = std::min(1.0, (1.0 - w) * g + w * c.g);
            b = std::min(1.0, (1.0 - w) * b + w * c.b);

            s = tree[rc].c;
            r = std::min(1.0, (1.0 - a) * r + a * s.r);
            g = std::min(1.0, (1.0 - a) * g + a * s.g);
            b = std::min(1.0, (1.0 - a) * b + a * s.b);

            tree[k].c.r = r;
            tree[k].c.g = g;
            tree[k].c.b = b;
        }


        r = std::min(255.0, 255.0 * tree[0].c.r);
        g = std::min(255.0, 255.0 * tree[0].c.g);
        b = std::min(255.0, 255.0 * tree[0].c.b);

        uchar4& pixel = payload.pixels[dst_i];
        pixel.x = static_cast<unsigned char>(r);
        pixel.y = static_cast<unsigned char>(g);
        pixel.z = static_cast<unsigned char>(b);
        pixel.w = static_cast<unsigned char>(0);

    }
}


__global__
void gpu_render(payload_t payload, size_t width, size_t height,
                shape_t* scene, size_t scene_size, floor_t floor,
                light_t* lights, size_t lights_size, size_t depth,
                double angle, vector3d_t cam, vector3d_t foc, tree_node_t* tree) {

    size_t offset = blockDim.x * gridDim.x;
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const double ratio = static_cast<double>(height) /
        static_cast<double>(width);
    const double dx = 2.0 / static_cast<double>(width - 1);
    const double dy = 2.0 / static_cast<double>(height - 1);
    const double dz = 1.0 / std::tan(angle / 2.0);

    const vector3d_t bz = (foc - cam).make_unit();
    const vector3d_t bx = (bz ^ vector3d_t::k()).make_unit();
    const vector3d_t by = (bx ^ bz).make_unit();


    const double floor_min_x = min(min(min(floor.rec.v1.x, floor.rec.v2.x), 
        floor.rec.v3.x), floor.rec.v4.x);
    const double floor_max_x = max(max(max(floor.rec.v1.x, floor.rec.v2.x), 
        floor.rec.v3.x), floor.rec.v4.x);
    const double floor_min_y = min(min(min(floor.rec.v1.y, floor.rec.v2.y), 
        floor.rec.v3.y), floor.rec.v4.y);
    const double floor_max_y = max(max(max(floor.rec.v1.y, floor.rec.v2.y), 
        floor.rec.v3.y), floor.rec.v4.y);

    double floor_width  = floor_max_x - floor_min_x;
    double floor_height = floor_max_y - floor_min_y;

    const double inf = 1e+6;
    const double eps = 1e-6;


    double r, g, b, a, w;
    intersection_t ret;
    vector3d_t pv;
    vector3d_t dv;
    vector3d_t nv;
    size_t ix;
    real_color_t c;
    real_color_t s;


    size_t k, rc, lc;
    size_t tree_size = 1 << depth;
    tree += index * tree_size;

    while(index < payload.size){
        size_t dst_i = payload.bindings[index].dst_i;
        size_t src_i = payload.bindings[index].src_i;
        size_t i = src_i % width, j = src_i / width;

        dv.x = (-1.0 + i * dx) * 1.000;
        dv.y = (+1.0 - j * dy) * ratio;
        dv.z = dz;
        dv = dv.transform(bx, by, bz);
        ret = intersect(cam, dv, 1.000, inf, scene, scene_size);
        tree[0].alive = ret.hit;

        if (tree[0].alive) {
            pv = cam + ret.t * dv;
            tree[0].dv = dv.make_unit();
            tree[0].pv = pv;
            tree[0].ix = ret.i;
            tree[0].nv = scene[ret.i].t.normal();
        }
        k = 0;
        while ((2 * k + 2) < tree_size) {
            lc = 2 * k + 1;
            rc = 2 * k + 2;
            if (!tree[k].alive) {
                tree[lc].alive = false;
                tree[rc].alive = false;
                ++k;
                continue;
            }
            dv = tree[k].dv;
            pv = tree[k].pv;
            nv = tree[k].nv;

            ret = intersect(pv, dv, eps, inf, scene, scene_size);
            tree[rc].alive = ret.hit;
            if (ret.hit) {
                tree[rc].ix = ret.i;
                tree[rc].nv = scene[ret.i].t.normal();
                tree[rc].pv = pv + ret.t * dv;
                tree[rc].dv = dv;
            }

            dv = (-dv).reflect(nv);

            ret = intersect(pv, dv, eps, inf, scene, scene_size);
            tree[lc].alive = ret.hit;
            if (tree[lc].alive) {
                tree[lc].ix = ret.i;
                tree[lc].nv = scene[ret.i].t.normal();
                tree[lc].pv = pv + ret.t * dv;
                tree[lc].dv = dv;
            }
            ++k;
        }

        while (k > 0) {
            --k;
            lc = 2 * k + 1;
            rc = 2 * k + 2;
            if (!tree[k].alive) {
                tree[k].c.r = 0;
                tree[k].c.g = 0;
                tree[k].c.b = 0;
                continue;
            }


            dv = tree[k].dv;
            pv = tree[k].pv;
            nv = tree[k].nv;
            ix = tree[k].ix;
            c = get_color(floor, scene, ix, pv,
                floor_min_x, floor_max_x,
                floor_min_y, floor_max_y);
            s = compute_lighting(pv, nv, -dv, ix,
                scene, scene_size, 
                lights, lights_size);

            a = c.a;
            w = scene[ix].ref;


            r = min(1.0, c.r * s.r);
            g = min(1.0, c.g * s.g);
            b = min(1.0, c.b * s.b);

            if (rc >= tree_size) {
                tree[k].c.r = r;
                tree[k].c.g = g;
                tree[k].c.b = b;
                continue;
            }

            c = tree[lc].c;
            r = min(1.0, (1 - w) * r + w * c.r);
            g = min(1.0, (1 - w) * g + w * c.g);
            b = min(1.0, (1 - w) * b + w * c.b);

            s = tree[rc].c;
            r = min(1.0, (1 - a) * r + a * s.r);
            g = min(1.0, (1 - a) * g + a * s.g);
            b = min(1.0, (1 - a) * b + a * s.b);

            tree[k].c.r = r;
            tree[k].c.g = g;
            tree[k].c.b = b;
        }


        r = min(255.0, 255.0 * tree[0].c.r);
        g = min(255.0, 255.0 * tree[0].c.g);
        b = min(255.0, 255.0 * tree[0].c.b);

        uchar4& pixel = payload.pixels[dst_i];
        pixel.x = static_cast<unsigned char>(r);
        pixel.y = static_cast<unsigned char>(g);
        pixel.z = static_cast<unsigned char>(b);
        pixel.w = static_cast<unsigned char>(0);
        index += offset;
    }
}



void cpu_ssaa(  uchar4* image,  size_t image_width, size_t image_height, 
                uchar4* ssaa,   size_t width_ratio, size_t height_ratio) {

    const size_t ssaa_width = image_width * width_ratio;
    auto at = [ ssaa_width, ssaa] (size_t i, size_t j) {
        return ssaa[j * ssaa_width + i];
    };
    const size_t block_size = width_ratio * height_ratio; 
    const size_t image_size = image_width * image_height;
#pragma omp parallel for
    for(int u = 0; u < image_size; ++u){
        size_t i = u % image_width;
        size_t j = u / image_width;
        size_t j_offset = j * height_ratio;
        size_t i_offset = i * width_ratio;
        double r = 0, g = 0, b = 0, w = 0;
        uchar4 pixel;
        for (size_t jj = 0; jj < height_ratio; ++jj) {
            for (size_t ii = 0; ii < width_ratio; ++ii) {
                pixel = at(  i_offset + ii, 
                             j_offset + jj);
                r += pixel.x;
                g += pixel.y;
                b += pixel.z;
                w += pixel.w;

            }
        }
        r /= static_cast<double>(block_size);
        g /= static_cast<double>(block_size);
        b /= static_cast<double>(block_size);
        w /= static_cast<double>(block_size);
        pixel.x = static_cast<unsigned char>(r);
        pixel.y = static_cast<unsigned char>(g);
        pixel.z = static_cast<unsigned char>(b);
        pixel.w = static_cast<unsigned char>(w);
        image[j * image_width + i] = pixel;
    }

}



__global__ 
void gpu_ssaa(  uchar4* image, size_t image_width, size_t image_height,
                uchar4* ssaa, size_t width_ratio, size_t height_ratio){


    size_t ssaa_width  = image_width * width_ratio;
    size_t block_size = width_ratio * height_ratio;
    size_t image_size = image_width * image_height;

    size_t offset = gridDim.x * blockDim.x;
    size_t index  = blockIdx.x * blockDim.x + threadIdx.x;

    while(index < image_size){
        size_t i = index % image_width;
        size_t j = index / image_width;
        double r = 0, g = 0, b = 0, w = 0;
        uchar4 pixel;
        for (size_t jj = 0; jj < height_ratio; ++jj) {
            for (size_t ii = 0; ii < width_ratio; ++ii) {
                pixel = ssaa[   (j * height_ratio + jj)
                                * ssaa_width + 
                                (i * width_ratio + ii)];
                r += static_cast<double>(pixel.x);
                g += static_cast<double>(pixel.y);
                b += static_cast<double>(pixel.z);
                w += static_cast<double>(pixel.w);

            }
        }
        r /= static_cast<double>(block_size);
        g /= static_cast<double>(block_size);
        b /= static_cast<double>(block_size);
        w /= static_cast<double>(block_size);
        pixel.x = static_cast<unsigned char>(r);
        pixel.y = static_cast<unsigned char>(g);
        pixel.z = static_cast<unsigned char>(b);
        pixel.w = static_cast<unsigned char>(w);
        image[j * image_width + i] = pixel;
        index += offset;
    }
}


enum class ComputingDevice {
    CPU,
    GPU
};
int MPI_Bcast(std::string& str, int root, MPI_Comm comm) {
    int rank, retval;
    retval = MPI_Comm_rank(comm, &rank);
    if (retval != MPI_SUCCESS) {
        return retval;
    }
    int size = static_cast<int>(str.size());
    retval = MPI_Bcast(&size, 1, MPI_INT, root, comm);
    if (retval != MPI_SUCCESS) {
        return retval;
    }
    char* buf = new char[size * sizeof(char)];
    if (rank == root) {
        std::copy(str.begin(), str.end(), buf);
    }
    retval = MPI_Bcast(buf, size, MPI_CHAR, root, comm);
    if (retval != MPI_SUCCESS) {
        delete[](buf);
        return retval;
    }
    if (rank != root) {
        str.assign(buf, buf + size);
    }
    delete[](buf);
    return MPI_SUCCESS;
}


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    ComputingDevice device;

    if (argc < 2) {
        device = ComputingDevice::GPU;
    }
    else if (argc == 2 && std::string(argv[1]) == "--default") {
        std::cout <<
            "400                                     \n"
            "img_ % d.data                           \n"
            "3840 2160 100                           \n"
            "7.0 3.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0 \n"
            "2.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0 \n"
            "                                        \n"
            "                                        \n"
            "0.0 - 1.5 0.0                           \n"
            "1.0 0.0 0.0                             \n"
            "1.5                                     \n"
            "0.3 0.4                                 \n"
            "3                                       \n"
            "                                        \n"
            "2.0 1.5 0.0                             \n"
            "0.0 1.0 0.0                             \n"
            "1.5                                     \n"
            "0.6 0.3                                 \n"
            "2                                       \n"
            "                                        \n"
            "                                        \n"
            "- 2.0 1.5 0.0                           \n"
            "0.0 0.7 0.7                             \n"
            "1.5                                     \n"
            "0.5 0.4                                 \n"
            "2                                       \n"
            "                                        \n"
            "- 10.0 - 10.0 - 1.5                     \n"
            "- 10.0 10.0 - 1.5                       \n"
            "10.0 10.0 - 1.5                         \n"
            "10.0 - 10.0 - 1.5                       \n"
            "texture.data                            \n"
            "1.0 1.0 1.0 0.3                         \n"
            "                                        \n"
            "                                        \n"
            "1                                       \n"
            "                                        \n"
            "- 10 - 10 3                             \n"
            "0.2 0.2 0.2                             \n"
            "5 4                                     \n";
            return 0;
    }
    else if (argc == 2 && std::string(argv[1]) == "--gpu") {
        device = ComputingDevice::GPU;
    }
    else if (argc == 2 && std::string(argv[1]) == "--cpu") {
        device = ComputingDevice::CPU;
    }
    else {
        std::cerr << "--gpu     : use gpu as computing device\n";
        std::cerr << "--cpu     : use cpu as computing device\n";
        std::cerr << "--default : show optimal configuration\n";
        return -1;
    }
    
    // input parameters
    int frames;
    std::string path;
    size_t path_specifier;

    int image_width, image_height;
    double view_angle;

    double  r0c, z0c, phi0c,
            arc, azc,
            wrc, wzc, wphic,
            prc, pzc;

    double  r0n, z0n, phi0n,
            arn, azn,
            wrn, wzn, wphin,
            prn, pzn;


    vector3d_t origin1;
    real_color_t c1;
    double r1, rc1, ln1;

    vector3d_t origin2;
    real_color_t c2;
    double r2, rc2, ln2;

    vector3d_t origin3;
    real_color_t c3;
    double r3, rc3, ln3;

    floor_t floor;
    double floor_reflection;

    std::string texture_path;

    int light_sources;
    std::vector<light_t> lights;

    int depth, ssaa_ratio;

    if (world_rank == 0) {
        // reading input parameters
        std::cin    >> frames;
        std::cin    >> path;

        std::cin    >> image_width >> image_height;

        std::cin    >> view_angle;

        std::cin    >> r0c >> z0c >> phi0c 
                    >> arc >> azc 
                    >> wrc >> wzc >> wphic 
                    >> prc >> pzc;

        std::cin    >> r0n >> z0n >> phi0n 
                    >> arn >> azn 
                    >> wrn >> wzn >> wphin 
                    >> prn >> pzn;

        std::cin    >> origin1.x >> origin1.y >> origin1.z
                    >> c1.r >> c1.g >> c1.b
                    >> r1   >> rc1  >> c1.a >> ln1;

        std::cin    >> origin2.x >> origin2.y >> origin2.z
                    >> c2.r >> c2.g >> c2.b
                    >> r2   >> rc2  >> c2.a >> ln2;

        std::cin    >> origin3.x >> origin3.y >> origin3.z
                    >> c3.r >> c3.g >> c3.b
                    >> r3   >> rc3  >> c3.a >> ln3;

        std::cin    >> floor.rec.v1.x >> floor.rec.v1.y >> floor.rec.v1.z
                    >> floor.rec.v2.x >> floor.rec.v2.y >> floor.rec.v2.z
                    >> floor.rec.v3.x >> floor.rec.v3.y >> floor.rec.v3.z
                    >> floor.rec.v4.x >> floor.rec.v4.y >> floor.rec.v4.z;

        std::cin    >> texture_path;

        std::cin    >> floor.col.r >> floor.col.g 
                    >> floor.col.b >> floor_reflection;
        floor.col.a = 1.0;

        std::cin >> light_sources;
        lights.resize(light_sources);
        for (size_t i = 0; i < light_sources; ++i) {
            lights[i].c.a = 1.0;
            lights[i].t = light_kind::point;
            std::cin >> lights[i].p.x;
            std::cin >> lights[i].p.y;
            std::cin >> lights[i].p.z;
            std::cin >> lights[i].c.r;
            std::cin >> lights[i].c.g;
            std::cin >> lights[i].c.b;
        }
        std::cin >> depth >> ssaa_ratio;

    
        {
            std::fstream fin(texture_path, std::ios::in | std::ios::binary);
            if (!fin.is_open()) {
                std::cerr << "Could not open texture file\n";
                return -1;
            }
            fin.read(reinterpret_cast<char*>(&floor.tex.width), sizeof(int));
            fin.read(reinterpret_cast<char*>(&floor.tex.height), sizeof(int));

            int size = floor.tex.width * floor.tex.height * sizeof(uchar4);
            floor.tex.data = reinterpret_cast<uchar4*>(operator new(size));
            fin.read(reinterpret_cast<char*>(floor.tex.data), size);

        }
    }

    MPI_Bcast(&frames,       1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&image_width , 1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&image_height, 1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&view_angle,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&floor.tex.width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.tex.height,1, MPI_INT, 0, MPI_COMM_WORLD);
    {
        int size = floor.tex.width * floor.tex.height * sizeof(uchar4);
        if (world_rank != 0) {
            floor.tex.data = reinterpret_cast<uchar4*>(operator new(size));
        }
        MPI_Bcast(floor.tex.data, size, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(&floor.col.r,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.col.g,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.col.b,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.col.a,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.rec.v1.x,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.rec.v1.y,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.rec.v1.z,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.rec.v2.x,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.rec.v2.y,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.rec.v2.z,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.rec.v3.x,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.rec.v3.y,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.rec.v3.z,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.rec.v4.x,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.rec.v4.y,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor.rec.v4.z,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_reflection, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&r0c,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&z0c,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&phi0c, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&arc,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&azc,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wrc,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wzc,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wphic, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&prc,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pzc,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&r0n,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&z0n,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&phi0n, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&arn,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&azn,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wrn,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wzn,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wphin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&prn,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pzn,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&origin1.x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&origin1.y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&origin1.z, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&origin2.x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&origin2.y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&origin2.z, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&origin3.x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&origin3.y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&origin3.z, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&c1.r, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c1.g, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c1.b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c1.a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&r1,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rc1,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ln1,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&c2.r, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c2.g, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c2.b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c2.a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&r2,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rc2,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ln2,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&c3.r, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c3.g, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c3.b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c3.a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&r3,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rc3,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ln3,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&light_sources, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0) {
        lights.resize(light_sources);
    }
    for (int i = 0; i < light_sources; ++i) {
        lights[i].t = light_kind::point;
        MPI_Bcast(&lights[i].p.x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lights[i].p.y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lights[i].p.z, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lights[i].c.r, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lights[i].c.g, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lights[i].c.b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lights[i].c.a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Bcast(path,         0, MPI_COMM_WORLD);
    MPI_Bcast(texture_path, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ssaa_ratio, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&depth,      1, MPI_INT, 0, MPI_COMM_WORLD);

    path_specifier = path.find("%d");
    if (path_specifier == std::string::npos) {
        std::cerr << "%d missing specifier in path!\n";
        return -1;
    }
    
    lights.push_back(
        light_t{
            light_kind::ambient,
            vector3d_t{0,0,0},
            real_color_t{.1,.1,.1}
        }
    );
    // calculating constant values
    const double pi = std::acos(-1);
    const double dt = 2.0 * pi / static_cast<double>(frames);
    view_angle = pi * view_angle / 180;

    const int ssaa_width   = image_width * ssaa_ratio;
    const int ssaa_height  = image_height * ssaa_ratio;
    const int ssaa_block   = ssaa_ratio * ssaa_ratio;

    const int image_pixels = image_width * image_height;
    const int full_load = image_pixels / world_size;
    const int rest_load = image_pixels % world_size;
    const int load_width = full_load + (world_rank < rest_load);
    const int load_height = 1;
    const int load_pixels = load_height * load_width;

    const int ssaa_load_width  = ssaa_ratio * load_width;
    const int ssaa_load_pixels = ssaa_block * load_pixels;

   
    std::string part1 = path.substr(0, path_specifier);
    std::string part2 = path.substr(path_specifier + 2);
    
    std::array<triangle_t, 4> s1 = tetrahedron();
    transform(s1, origin1, r1);

    std::array<triangle_t, 20> s2 = icosahedron();
    transform(s2, origin2, r2);

    std::array<triangle_t, 36> s3 = dodecahedron();
    transform(s3, origin3, r3);

    std::set<std::pair<vector3d_t, vector3d_t>> edges1;
    for (size_t i = 0; i < s1.size(); ++i) {
        triangle_t& t = s1[i];
        edges1.insert(std::make_pair(t.v1, t.v2));
        edges1.insert(std::make_pair(t.v1, t.v3));
        edges1.insert(std::make_pair(t.v2, t.v3));
    }

    std::set<std::pair<vector3d_t, vector3d_t>> edges2;
    for (size_t i = 0; i < s2.size(); ++i) {
        triangle_t& t = s2[i];
        edges2.insert(std::make_pair(t.v1, t.v2));
        edges2.insert(std::make_pair(t.v1, t.v3));
        edges2.insert(std::make_pair(t.v2, t.v3));
    }

    std::set<std::pair<vector3d_t, vector3d_t>> edges3;
    for (size_t i = 0; i < s3.size(); ++i) {
        triangle_t& t = s3[i];
        edges3.insert(std::make_pair(t.v1, t.v2));
        edges3.insert(std::make_pair(t.v1, t.v3));
        edges3.insert(std::make_pair(t.v2, t.v3));
    }

    std::set<vector3d_t> diodes1;
    for (auto it = edges1.begin(); it != edges1.end(); ++it) {
        vector3d_t dv = it->second - it->first;
        double dk = 1.0 / static_cast<double>(ln1 - 1);
        for (size_t i = 0; i < ln1; ++i) {
            diodes1.insert(it->first + i * dk * dv);
        }
    }
    for (auto it = diodes1.begin(); it != diodes1.end(); ++it) {
        lights.push_back(
            light_t{
                light_kind::diode,
                *it,
                {0.002,0.002,0.002}
            }
        );
    }
    std::set<vector3d_t> diodes2;
    for (auto it = edges2.begin(); it != edges2.end(); ++it) {
        vector3d_t dv = it->second - it->first;
        double dk = 1.0 / static_cast<double>(ln2 - 1);
        for (size_t i = 0; i < ln2; ++i) {
            diodes2.insert(it->first + i * dk * dv);
        }
    }
    for (auto it = diodes2.begin(); it != diodes2.end(); ++it) {
        lights.push_back(
            light_t{
                light_kind::diode,
                *it,
                {0.002,0.002,0.002}
            }
        );
    }
    std::set<vector3d_t> diodes3;
    for (auto it = edges3.begin(); it != edges3.end(); ++it) {
        vector3d_t dv = it->second - it->first;
        double dk = 1.0 / static_cast<double>(ln3 - 1);
        for (size_t i = 0; i < ln3; ++i) {
            diodes3.insert(it->first + i * dk * dv);
        }
    }
    for (auto it = diodes3.begin(); it != diodes3.end(); ++it) {
        lights.push_back(
            light_t{
                light_kind::diode,
                *it,
                {0.002,0.002,0.002}
            }
        );
    }


    std::vector<shape_t> scene;
    scene.reserve(s1.size() + s2.size() + s3.size() + 2);
    for (size_t i = 0; i < s1.size();) {
        scene.push_back(shape_t{c1, rc1, false, s1[i++]});
    }
    for (size_t i = 0; i < s2.size();) {
        scene.push_back(shape_t{c2, rc2, false, s2[i++]});
    }
    for (size_t i = 0; i < s3.size();) {
        scene.push_back(shape_t{ c3, rc3, false, s3[i++] });
        scene.push_back(shape_t{ c3, rc3, false, s3[i++] });
        scene.push_back(shape_t{ c3, rc3, false, s3[i++] });
    }
    
    scene.push_back(
        shape_t{
            floor.col, floor_reflection, true, 
            triangle_t{floor.rec.v2, floor.rec.v1, floor.rec.v3},
            (int)scene.size() - 0
        }
    );
    scene.push_back(
        shape_t{
            floor.col, floor_reflection, true,
            triangle_t{floor.rec.v3, floor.rec.v1, floor.rec.v4},
            (int)scene.size() - 1
        }
    );

    auto cam_cart = [r0c, z0c, phi0c, arc, azc, wrc, wzc, wphic, prc, pzc](double t) {
        double r = r0c + arc * std::sin(wrc * t + prc);
        double z = z0c + azc * std::sin(wzc * t + pzc);
        double phi = phi0c + wphic * t;
        return vector3d_t{ r * std::cos(phi), r * std::sin(phi), z };
    };

    auto foc_cart = [r0n, z0n, phi0n, arn, azn, wrn, wzn, wphin, prn, pzn](double t) {
        double r = r0n + arn * std::sin(wrn * t + prn);
        double z = z0n + azn * std::sin(wzn * t + pzn);
        double phi = phi0n + wphin * t;
        return vector3d_t{ r * std::cos(phi), r * std::sin(phi), z };
    };

    
    std::vector<uchar4> image(load_pixels);
    std::vector<binding_t> bindings(ssaa_load_pixels);

    for (int u = 0; u < ssaa_load_pixels; ++u) {

        int lbk = u / ssaa_block;
        int gbk = world_rank + world_size * lbk;
        int gbi = gbk % image_width;
        int gbj = gbk / image_width;

        int lpk = u % ssaa_block;
        int lpi = lpk % ssaa_ratio;
        int lpj = lpk / ssaa_ratio;

        int gpi = ssaa_ratio * gbi + lpi;
        int gpj = ssaa_ratio * gbj + lpj;

        bindings[u].src_i = gpj * ssaa_width + gpi;
        bindings[u].dst_i = ssaa_load_width * lpj +
            lbk * ssaa_ratio + lpi;
    }

    std::vector<MPI_Aint> displacements(load_pixels);
    std::vector<int> blocklengths(load_pixels, 4);
    for (int i = 0; i < load_pixels; ++i) {
        displacements[i] = 4 * (world_rank + i * world_size);
    }
    MPI_Datatype mpi_file_view;
    MPI_Type_create_hindexed(load_pixels, blocklengths.data(),
        displacements.data(), MPI_CHAR, &mpi_file_view);
    MPI_Type_commit(&mpi_file_view);
    MPI_Offset mpi_offset = 2 * sizeof(int);
    MPI_File mpi_file;
    MPI_Status mpi_status;
 
    std::chrono::high_resolution_clock::time_point tic;
    std::chrono::high_resolution_clock::time_point tac;

    payload_t payload;
    payload.size = ssaa_load_pixels;
    if (device == ComputingDevice::CPU) { 
        std::vector<uchar4> ssaa(ssaa_load_pixels);
        
        payload.bindings = bindings.data();
        payload.pixels = ssaa.data();


        for (size_t f = 0; f < frames; ++f) {
            double t = f * dt;
            tic = std::chrono::high_resolution_clock::now();
            cpu_render( payload, ssaa_width, ssaa_height,
                        scene.data(), scene.size(), floor,
                        lights.data(), lights.size(), depth,
                        view_angle, cam_cart(t), foc_cart(t));
            cpu_ssaa(   image.data(), load_width, load_height,
                        ssaa.data() , ssaa_ratio , ssaa_ratio);
            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);
            tac = std::chrono::high_resolution_clock::now();

            std::string fname = part1 + std::to_string(f) + part2;

            MPI_File_delete(fname.c_str(), MPI_INFO_NULL);
            MPI_File_open(MPI_COMM_WORLD, fname.c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, 
                &mpi_file);
            if (world_rank == 0) {
                MPI_File_write_at(mpi_file, 0 * sizeof(int), 
                    &image_width, 1, MPI_INT, &mpi_status);
                MPI_File_write_at(mpi_file, 1 * sizeof(int),
                    &image_height, 1, MPI_INT, &mpi_status);
            }

            MPI_File_set_view(mpi_file, mpi_offset, MPI_CHAR,
                mpi_file_view, "native", MPI_INFO_NULL);
            MPI_File_write_all(mpi_file, image.data(),
                4 * load_pixels , MPI_CHAR, &mpi_status);

            MPI_File_close(&mpi_file);

            if (world_rank == 0) {
                std::cerr   << "Sample [ "  << std::setw(5) 
                            << f << " / "   << std::setw(5) 
                            << frames << " ]" << " elapsed: "
                            << std::chrono::duration_cast<
                                std::chrono::milliseconds>
                            (tac - tic).count() << " ms\n";
            }
                    
        }
    }
    else {
        uchar4* gpu_image, *gpu_ssaa_mem;

        shape_t* gpu_scene;
        light_t* gpu_lights;
        uchar4* gpu_texture;
        tree_node_t* gpu_tree;
        binding_t *gpu_bindings;

        int device_count;
        HANDLE_ERROR(cudaGetDeviceCount(&device_count));
        HANDLE_ERROR(cudaSetDevice(world_rank % device_count));

        size_t tex_size = floor.tex.width * floor.tex.height;
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&gpu_image), load_pixels           * sizeof(uchar4)     ));
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&gpu_ssaa_mem), ssaa_load_pixels   * sizeof(uchar4)     ));
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&gpu_texture), tex_size            * sizeof(uchar4)     ));
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&gpu_scene), scene.size()          * sizeof(shape_t)    ));
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&gpu_lights), lights.size()        * sizeof(light_t)    ));
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&gpu_bindings), bindings.size()    * sizeof(binding_t)  ));
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&gpu_tree), 256*256*(1<<depth)      * sizeof(tree_node_t)));

        payload.bindings = gpu_bindings;
        payload.pixels   = gpu_ssaa_mem;

        HANDLE_ERROR(cudaMemcpy(gpu_texture, floor.tex.data, tex_size          * sizeof(uchar4),    cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(gpu_scene, scene.data(), scene.size()          * sizeof(shape_t),   cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(gpu_lights, lights.data(), lights.size()       * sizeof(light_t),   cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(gpu_bindings, bindings.data(), bindings.size() * sizeof(binding_t), cudaMemcpyHostToDevice));

        floor_t gpu_floor_obj = floor;
        gpu_floor_obj.tex.data = gpu_texture;
        for (size_t f = 0; f < frames; ++f) {
            double t = f * dt;
            tic = std::chrono::high_resolution_clock::now();
            gpu_render<<<256, 256>>> (   payload, ssaa_width, ssaa_height,
                                        gpu_scene, scene.size(), gpu_floor_obj,
                                        gpu_lights, lights.size(), depth,
                                        view_angle, cam_cart(t), foc_cart(t), gpu_tree);
            HANDLE_ERROR(cudaDeviceSynchronize());
            HANDLE_ERROR(cudaGetLastError());
            gpu_ssaa <<<256, 256 >>> (gpu_image, load_width, load_height,
                                      gpu_ssaa_mem, ssaa_ratio, ssaa_ratio);
            HANDLE_ERROR(cudaDeviceSynchronize());
            HANDLE_ERROR(cudaGetLastError());
            HANDLE_ERROR(cudaMemcpy(image.data(), gpu_image, load_pixels * sizeof(uchar4), cudaMemcpyDeviceToHost));
            MPI_Barrier(MPI_COMM_WORLD);
            tac = std::chrono::high_resolution_clock::now();

            std::string fname = part1 + std::to_string(f) + part2;

            MPI_File_delete(fname.c_str(), MPI_INFO_NULL);
            MPI_File_open(MPI_COMM_WORLD, fname.c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                &mpi_file);
            if (world_rank == 0) {
                MPI_File_write_at(mpi_file, 0 * sizeof(int),
                    &image_width, 1, MPI_INT, &mpi_status);
                MPI_File_write_at(mpi_file, 1 * sizeof(int),
                    &image_height, 1, MPI_INT, &mpi_status);
            }

            MPI_File_set_view(mpi_file, mpi_offset, MPI_CHAR,
                mpi_file_view, "native", MPI_INFO_NULL);
            MPI_File_write_all(mpi_file, image.data(),
                4 * load_pixels, MPI_CHAR, &mpi_status);

            MPI_File_close(&mpi_file);
            if (world_rank == 0) {
                std::cerr << "Sample [ " << std::setw(5)
                    << f << " / " << std::setw(5)
                    << frames << " ]" << " elapsed: "
                    << std::chrono::duration_cast<
                    std::chrono::milliseconds>
                    (tac - tic).count() << " ms\n";

            }
        }

        cudaFree(gpu_image   );
        cudaFree(gpu_ssaa_mem);
        cudaFree(gpu_texture );
        cudaFree(gpu_scene   );
        cudaFree(gpu_lights  );
        cudaFree(gpu_tree    );
        cudaFree(gpu_bindings);
    }
    delete[](floor.tex.data);
    MPI_Type_free(&mpi_file_view);
    MPI_Finalize();

    return 0;
}
