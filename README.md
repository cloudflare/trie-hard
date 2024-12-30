[![Crates.io](https://img.shields.io/crates/v/trie-hard.svg)](https://crates.io/crates/trie-hard)
[![Documentation](https://docs.rs/trie-hard/badge.svg)](https://docs.rs/trie-hard/)

# Don't just Trie, Trie Hard

This crate is an implementation of the [trie](https://en.wikipedia.org/wiki/Trie) data structure that is optimized for reading from small maps where a large number of misses are expected. This is still a work in progress in the sense that as none of the other features you would expect to find in a trie (like prefix search), but it is used in production in Cloudflare's [Pingora](https://blog.cloudflare.com/pingora-open-source) to detect and remove specific headers from 30 million requests every second before they are proxied to their final destinations.

## Performance

There are several other trie implementations for rust that are more full-featured, so if you are looking for a more robust tool, you will probably want to check out [`radix_trie`](https://crates.io/crates/radix_trie) which seems to have the best features and performance. On the other hand, if you want raw speed and have the same narrow use case, you came to the right place!

Here is a chart showing the time taken to read 10k entries from a map that consists of 119 entries containing only lower-case characters, numbers, and `-`. As you can see, when miss rate gets above 50% the performance of trie-hard surpasses `std::HashMap` and improves as miss rates get higher. 

![Trie Hard read is faster than HashMap for small maps where miss rate is high](https://github.com/cloudflare/trie-hard/blob/main/resources/HeadersVsHashMap.png?raw=true "Header Read vs HashMap Benchmark")

The improvement of performance as miss rate grows is one of the features of tries, and you can see the same trend in the performance of `radix-trie`. 

![Trie Hard read is faster than RadixTrie but follows the same curve](https://github.com/cloudflare/trie-hard/blob/main/resources/HeadersVsRadixTrie.png?raw=true "Header Read vs RadixTrie Benchmark")

This also shows the difference in performance between radix-trie and trie-hard, but this should not be interpreted as a reason to use trie-hard over radix-trie. radix-trie is a full-featured trie and balances the read performance with write performance. For fairness, here is a breakdown of the loading performance of the two trie implementations.

| Item       | Time to load 15.5k words |
| ---------- | ------------------------ |
| Trie Hard  | 11.92 ms                 |
| Radix Trie | 3.49 ms                  |

For insertion radix is ~3x times faster and also supports incremental changes whereas trie-hard is only designed for bulk loading.

## How Does it Work?

Trie Hard achieves its speed in 2 ways.

1. All node and edge information is kept in contiguous regions on the heap. This prevents jumping around in memory during gets, and maximizes the chance of child nodes already appearing in cache.
2. Relationships between nodes and edges is encoded into individual bits in unsigned integers.

### Example

Let's say we want to store the (uninteresting) strings, "and", "ant", "dad", "do", "dot" in our trie. Let's work through an example of how the data is organized and how we can read from it.

#### Construction

Trie-hard only supports bulk loading so, we run 

```rust
let trie = ["and", "ant", "dad", "do", "dot"].into_iter().collect::<TrieHard<'_, _>>();
```

The first thing trie-hard does is find all the used bytes in the entries. It assigns each a unique 1-bit mask. For our simple case, this will look like

| Byte   | Mask      |
| ------ | --------- |
| `b'a'` | `0b00001` |
| `b'd'` | `0b00010` |
| `b'n'` | `0b00100` |
| `b'o'` | `0b01000` |
| `b't'` | `0b10000` |

The number of masks required determines the underlying integer type used to represent the mask. In our case, we only have 5 bits, so the underlying type will be `u8`. The implication of needing a bit for each unique byte is that potentially we would require a 256-bit integer (`u256`), so an implementation of `U256` is provided as part of this crate. _Note_ The `U256` type should not be used directly. It only implements (and tests) the few operations needed to work with `trie-hard`. The use case `trie-hard` was designed for only needs to support at most 37 unique bytes, but in practice only 30 unique bytes appear in the stored map. This means we are storing our underlying tree information in `u32`s.

Next we will construct the graph representing the tree starting with the bytes that appear first in the input strings. Only `a` and `d` appear in the first position, so for the node representing the first byte (and also the root of the trie), we create a mask indicating only `a` and `d` are allowed.

```rust
let root = Node {
    // a (0b00001) + d (0b00010) = 0b00011
    mask: 0b11,
    // ..
}
```

This tells us that if a byte other than `a` or `d` appears in the first position, the key being tested does not appear in the trie. This ability to make an exclusion decision at every step is what makes tries more appealing than even hashmaps in some cases. Searching for a string in a hashmap requires hashing the entire string whereas a trie can potentially determine that a string is not part of a set within a single byte.

If the byte is `a` or `d` we still need to know which node to go to next. All nodes in the graph are stored in a contiguous vector (with the root node at index zero). Each node will contain the information on where its child appears in the array of nodes. In our example the root node will point to nodes with indexes 1 and 2. Where 1 is the index with keys starting with `a` and 2 is the node for keys starting with `d`. It is important that these child nodes are ordered by their corresponding byte. 

```rust
let root = Node {
    mask: 0b11,
    // 1 -> keys that start with b'a'
    // 2 -> keys that start with b'd'
    children: vec![1, 2],
    // ..
}
```

> _Note_. In the final map, the child indices are not stored in per-node vecs. They are instead stored in a central vec, and only the index into that vec corresponding with this node is stored in the node.

At this point we can visualize the conceptual trie and trie-hard like this

| Conceptual                                                                                                                                                    | Trie-Hard                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![First Layer Conceptual Trie](https://github.com/cloudflare/trie-hard/blob/main/resources/FirstLayerVanilla.png?raw=true "Header Read vs HashMap Benchmark") | ![Trie Hard read is faster than HashMap for small maps where miss rate is high](https://github.com/cloudflare/trie-hard/blob/main/resources/FirstLayerTrieHard.png?raw=true "Header Read vs HashMap Benchmark") |

Because of the recursive nature of a trie, we can repeat the same process of creating a mask based on allowed bytes at each node and preparing a set of children for each node. When we reach a complete word that appears in the initial set, we need to signify that the node is a valid word. Visually we will mark them with green, but in rust they just appear as a different enum variant of `TrieNode`.

After repeating for one more layer, we can visualize the trie like the this.

| Conceptual                                                                                                                                                     | Trie-Hard                                                                                                                                                                                                        |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![First Layer Conceptual Trie](https://github.com/cloudflare/trie-hard/blob/main/resources/SecondLayerVanilla.png?raw=true "Header Read vs HashMap Benchmark") | ![Trie Hard read is faster than HashMap for small maps where miss rate is high](https://github.com/cloudflare/trie-hard/blob/main/resources/SecondLayerTrieHard.png?raw=true "Header Read vs HashMap Benchmark") |

Notice that `do` shows up as green because it is a complete word found in the original collection.

Finally, we add the last layer and complete this small trie.


| Conceptual                                                                                                                                          | Trie-Hard                                                                                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![First Layer Conceptual Trie](https://github.com/cloudflare/trie-hard/blob/main/resources/Vanilla.png?raw=true "Header Read vs HashMap Benchmark") | ![Trie Hard read is faster than HashMap for small maps where miss rate is high](https://github.com/cloudflare/trie-hard/blob/main/resources/TrieHard.png?raw=true "Header Read vs HashMap Benchmark") |

#### Reading 

When reading from the trie, we are checking a key to see if it is contained in the trie. As stated earlier, the benefit of a trie is that it lets lookups fail as fast as possible. Let's walk through the process of reading from trie-hard to see why. 

Trie's work on a per-byte basis, so each step of the read process takes a single byte from the input key, and checks the current node in the trie to see if is allowed. The first thing we do with the input byte is convert it from its `u8` value into its mask using the lookup table we created while constructing the trie. Let's say we want to do a lookup `dot`. We start with the first byte `d` and the current (root) node. 

`d` converts to a mask of `00010`. To determine if `d` is allowed in the root node, we take a bit-wise `&` of their masks. If the result > 0, then the byte is allowed. The mask for the root node is `00011`, and `00011 & 00010 = 00010 > 0`, so `d` is allowed. 

Next we need to determine which node to traverse for our input. This involves some more bit-wise arithmetic in the form of this **_one weird trick_**. We can count the number of set bits in the node's mask that are less-significant than the set bit in the input byte's mask. The number of set bits gives us the index in to the array of children. The formula/rust code for this is below. Notice it uses all bit-wise operations and intrinsics, so this operation equates to just a few cpu instructions in the compiled code.

```rust
let child_index = ((input_mask - 1) & node.mask).count_ones()
```

Applying this formula to our current input mask and node, we get

```rust
//            ((input_mask - 1) & node.mask).count_ones()              
child_index = ((  0b00010  - 1) &  0b00011 ).count_ones() // = 1
```

The next node is the one at **index = 1** in the current node's child array.

The reason this works may actually be easier to understand with a larger example. Consider this node from a larger trie (unrelated to our original example).

```rust
TrieNode {
    // bits for - n   h  fd  a
    mask: 0b000000100010011001,
    // 5 possible children for 5 set bits in mask
    children: [
        3,  // <-- a
        6,  // <-- d
        9,  // <-- f
        10, // <-- h
        12  // <-- n
    ]
}
```

If we want to know which child to go to after receiving the byte `h`, we first need `h`'s mask = `000000000010000000`. Now if we step through the formula we see that subtracting 1 from the `h`'s mask gives us:

```
    000000000010000000
  -                  1
 ----------------------
    000000000001111111
```
Notice this gives us a new mask where _all_ bit with lower significance than `h`'s are set. Continuing to follow the formula, we apply a big-wise `&` between the new mask and the node's mask, we get:

```

    000000100010011001
  & 000000000001111111
 ----------------------
    000000000000011001
```

The bits in this new mask are exactly what we wanted: only the set bits from the node's mask that have a lower significance than `h`'s. The last step is to call [`count_ones`](https://doc.rust-lang.org/std/primitive.u32.html#method.count_ones) which is provided as an intrinsic by llvm. This gives us the answer of **3**.

Going back to our original example, we now repeat the same steps for the next input byte and the new current node. This process continues until we run out of input or encounter a node that will not accept the current byte. 
